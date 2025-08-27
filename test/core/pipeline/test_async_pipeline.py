# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from haystack import AsyncPipeline, component
from haystack.core.errors import PipelineRuntimeError


def test_async_pipeline_reentrance(waiting_component, spying_tracer):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    run_data = [{"wait_for": 0.001}, {"wait_for": 0.002}]

    async def run_all():
        # Create concurrent tasks for each pipeline run
        tasks = [pp.run_async(data) for data in run_data]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run_async"]
    for span in component_spans:
        assert span.tags["haystack.component.visits"] == 1


def test_run_in_sync_context(waiting_component):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    result = pp.run({"wait_for": 0.001})

    assert result == {"wait": {"waited_for": 0.001}}


def test_run_in_async_context_raises_runtime_error():
    pp = AsyncPipeline()

    async def call_run():
        pp.run({})

    with pytest.raises(RuntimeError, match="Cannot call run\\(\\) from within an async context"):
        asyncio.run(call_run())


class TestAsyncPipelineErrorHandling:
    """Tests for error handling in AsyncPipeline."""

    def test_error_in_async_component(self):
        """Test error handling in async components."""

        @component
        class AsyncErrorComponent:
            __haystack_supports_async__ = True

            @component.output_types(output=str)
            async def run_async(self):
                raise ValueError("Async error")

            @component.output_types(output=str)
            def run(self):
                return {"output": "This should not be called"}

        error_component = AsyncErrorComponent()
        pp = AsyncPipeline()
        pp.add_component("async_error", error_component)

        with pytest.raises(PipelineRuntimeError) as exc_info:
            asyncio.run(pp.run_async({}))

        # Verify error contains component information
        error_str = str(exc_info.value)
        assert "Component name: 'async_error'" in error_str
        assert "Component type: 'AsyncErrorComponent'" in error_str
        assert "Async error" in error_str

    def test_error_in_non_async_component(self):
        """Test error handling in non-async components."""

        @component
        class NonAsyncErrorComponent:
            # We'll omit __haystack_supports_async__ entirely to guarantee it uses run()
            # This is the most realistic scenario for components without async support

            @component.output_types(output=str)
            def run(self):
                raise ValueError("Non-async error")

        error_component = NonAsyncErrorComponent()
        pp = AsyncPipeline()
        pp.add_component("non_async_error", error_component)

        with pytest.raises(PipelineRuntimeError) as exc_info:
            asyncio.run(pp.run_async({}))

        # Verify error contains component information
        error_str = str(exc_info.value)
        assert "Component name: 'non_async_error'" in error_str
        assert "Component type: 'NonAsyncErrorComponent'" in error_str
        assert "Non-async error" in error_str
