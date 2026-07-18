# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Meta functions for FlagOS device
# These are used by torch.compile and other meta-dispatch mechanisms


# You can add meta implementations here if needed
# For example:
# @impl("aten::some_op", "Meta")
# def some_op_meta(self):
#     return torch.empty_like(self)
