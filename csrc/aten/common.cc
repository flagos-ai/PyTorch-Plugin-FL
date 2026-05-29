// Copyright (c) 2026, BAAI. All rights reserved.

#include "common.h"

#include <cstdio>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace at::native::flagos {

namespace {

std::string DefaultConfigPath() {
#ifndef _WIN32
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(GetBackendForOp), &info) && info.dli_fname) {
    std::string lib_path(info.dli_fname);
    auto pos = lib_path.rfind('/');
    if (pos != std::string::npos) {
      std::string dir = lib_path.substr(0, pos);
      // Try package-relative: <dir>/../torch_fl/backends.conf
      std::string candidate = dir + "/../torch_fl/backends.conf";
      std::ifstream test(candidate);
      if (test.is_open()) return candidate;
      // Try: <dir>/backends.conf
      candidate = dir + "/backends.conf";
      test.open(candidate);
      if (test.is_open()) return candidate;
    }
  }
#endif
  // Fallback to build-time path
  return FLAGOS_SOURCE_ROOT "/torch_fl/backends.conf";
}

std::unordered_map<std::string, Backend> LoadBackendConfig() {
  std::unordered_map<std::string, Backend> table;

  const char* env = std::getenv("FLAGOS_BACKEND_CONFIG");
  std::string path = env ? env : DefaultConfigPath();

  std::ifstream f(path);
  if (!f.is_open()) {
    return table;
  }

  fprintf(stderr, "[flagos] loading backend config from %s\n", path.c_str());

  std::string line;
  while (std::getline(f, line)) {
    // strip comments
    auto comment = line.find('#');
    if (comment != std::string::npos) line = line.substr(0, comment);

    auto eq = line.find('=');
    if (eq == std::string::npos) continue;

    auto trim = [](std::string s) {
      size_t l = s.find_first_not_of(" \t\r\n");
      size_t r = s.find_last_not_of(" \t\r\n");
      return (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
    };

    std::string op = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));

    if (op.empty() || val.empty()) continue;

    if (val == "cuda") {
      table[op] = Backend::kCuda;
    } else if (val == "ascend") {
      table[op] = Backend::kAscend;
    } else if (val == "flagos" || val == "flaggems") {
      table[op] = Backend::kFlagOs;
    } else if (val == "flagos_python" || val == "flaggems_python") {
      table[op] = Backend::kFlagOsPython;
    } else {
      fprintf(stderr, "[flagos] unknown backend '%s' for op '%s', using flagos\n", val.c_str(), op.c_str());
      table[op] = Backend::kFlagOs;
    }
  }

  // Per-op env var overrides: FLAGOS_OP_<op_name>=cuda|flaggems
  // e.g. FLAGOS_OP_mm=cuda  or  FLAGOS_OP_mm__out=cuda
  // Dots in op names are replaced with double underscores to avoid ambiguity
  // with ops that already contain underscores (e.g. mm_out vs mm.out).
  for (auto& [op, _] : table) {
    std::string key = "FLAGOS_OP_";
    for (char c : op) {
      if (c == '.') key += "__";
      else key += c;
    }
    const char* override_val = std::getenv(key.c_str());
    if (!override_val) continue;
    std::string v(override_val);
    if (v == "cuda") {
      table[op] = Backend::kCuda;
      fprintf(stderr, "[flagos] env override: %s -> cuda\n", op.c_str());
    } else if (v == "ascend") {
      table[op] = Backend::kAscend;
      fprintf(stderr, "[flagos] env override: %s -> ascend\n", op.c_str());
    } else if (v == "flagos" || v == "flaggems") {
      table[op] = Backend::kFlagOs;
      fprintf(stderr, "[flagos] env override: %s -> flaggems\n", op.c_str());
    } else if (v == "flagos_python" || v == "flaggems_python") {
      table[op] = Backend::kFlagOsPython;
      fprintf(stderr, "[flagos] env override: %s -> flaggems_python\n", op.c_str());
    }
  }

  return table;
}

const std::unordered_map<std::string, Backend>& BackendTable() {
  static const auto table = LoadBackendConfig();
  return table;
}

} // namespace

Backend GetBackendForOp(const std::string& op_name) {
  const auto& table = BackendTable();
  auto it = table.find(op_name);
  return it != table.end() ? it->second : Backend::kFlagOs;
}

} // namespace at::native::flagos