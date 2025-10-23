# Copyright 2025 Google LLC
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

from __future__ import annotations

from .tool_context import ToolContext


def exit_sequence(tool_context: ToolContext):
    """Exits the sequential execution of agents immediately.

    Call this function when you encounter a terminal condition and want to
    prevent subsequent agents in the sequence from executing. This will also
    stop any remaining events from the current agent.

    This tool is specifically designed for use within SequentialAgent contexts.
    When called, it sets the escalate flag, which causes the SequentialAgent
    to terminate the sequence immediately, preventing both:
    - Subsequent events from the current sub-agent
    - All remaining sub-agents in the sequence

    Use cases:
    - A blocking error is encountered that makes further processing impossible
    - A definitive answer is found early, making subsequent agents unnecessary
    - A security or validation check fails and the workflow must stop
    - Resource limits are reached and safe termination is required

    Example:
      If you're in a sequence of [validator, processor, finalizer] agents,
      and the validator finds invalid data, it can call exit_sequence() to
      prevent the processor and finalizer from running on bad data.

    Args:
      tool_context: The context of the current tool invocation.
    """
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
