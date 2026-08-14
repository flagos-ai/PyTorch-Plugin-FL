// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <flagos.h>
#include <tops_runtime_api.h>

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int /*priority*/) {
  // The tops runtime has no stream priority support; honour the flags only.
  topsError_t err = topsStreamCreateWithFlags(
      reinterpret_cast<topsStream_t*>(stream), flags);
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  topsError_t err = topsStreamCreate(reinterpret_cast<topsStream_t*>(stream));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t /*stream*/, int* priority) {
  // No priority query in the tops runtime.
  *priority = 0;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  topsError_t err = topsStreamDestroy(reinterpret_cast<topsStream_t>(stream));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  topsError_t err = topsStreamQuery(reinterpret_cast<topsStream_t>(stream));
  if (err == topsSuccess) {
    return Success;
  } else if (err == topsErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  topsError_t err =
      topsStreamSynchronize(reinterpret_cast<topsStream_t>(stream));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  topsError_t err = topsStreamWaitEvent(
      reinterpret_cast<topsStream_t>(stream),
      reinterpret_cast<topsEvent_t>(event),
      flags);
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  topsError_t err = topsEventCreateWithFlags(
      reinterpret_cast<topsEvent_t*>(event), flags);
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  topsError_t err = topsEventCreate(reinterpret_cast<topsEvent_t*>(event));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  topsError_t err = topsEventDestroy(reinterpret_cast<topsEvent_t>(event));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  topsError_t err = topsEventRecord(
      reinterpret_cast<topsEvent_t>(event),
      reinterpret_cast<topsStream_t>(stream));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  topsError_t err = topsEventSynchronize(reinterpret_cast<topsEvent_t>(event));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  topsError_t err = topsEventQuery(reinterpret_cast<topsEvent_t>(event));
  if (err == topsSuccess) {
    return Success;
  } else if (err == topsErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  topsError_t err = topsEventElapsedTime(
      ms,
      reinterpret_cast<topsEvent_t>(start),
      reinterpret_cast<topsEvent_t>(end));
  return (err == topsSuccess) ? Success : ErrorUnknown;
}
