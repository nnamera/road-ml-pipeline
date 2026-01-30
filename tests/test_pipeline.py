"""Tests for pipeline module.

Copyright (c) 2024-2026 BlackRoad OS, Inc. All rights reserved.
"""

import pytest

from mlpipeline_core.pipeline import Pipeline, PipelineRun, PipelineStatus
from mlpipeline_core.pipeline.step import Step, StepConfig, StepStatus
from mlpipeline_core.pipeline.dag import DAG


class TestPipeline:
    """Test Pipeline class."""

    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = Pipeline(name="test-pipeline")
        assert pipeline.name == "test-pipeline"
        # Accessing private attribute as public property doesn't exist
        assert len(pipeline._steps) == 0

    def test_add_step(self):
        """Test adding steps."""
        pipeline = Pipeline(name="test-pipeline")

        @pipeline.step(name="step1")
        def step1():
            return {"value": 1}

        assert "step1" in pipeline._steps
        assert len(pipeline._steps) == 1

    def test_step_dependencies(self):
        """Test step dependencies."""
        pipeline = Pipeline(name="test-pipeline")

        @pipeline.step(name="step1")
        def step1():
            return {"value": 1}

        @pipeline.step(name="step2", depends_on=["step1"])
        def step2(step1):
            return {"result": step1["value"] * 2}

        dag = pipeline._dag
        assert "step1" in dag.get_dependencies("step2")

    def test_pipeline_run(self):
        """Test pipeline execution."""
        pipeline = Pipeline(name="test-pipeline")

        @pipeline.step(name="add")
        def add():
            return {"sum": 5}

        # The parameter name must match the dependency name
        @pipeline.step(name="multiply", depends_on=["add"])
        def multiply(add):
            return {"product": add["sum"] * 2}

        run = pipeline.run()
        assert run.status == PipelineStatus.COMPLETED
        assert "multiply" in run.step_results
        assert run.step_results["multiply"].output["product"] == 10


class TestStep:
    """Test Step class."""

    def test_step_creation(self):
        """Test step creation."""
        def fn():
            return {"x": 1}

        step = Step(name="test-step", func=fn)
        assert step.name == "test-step"

    def test_step_execution(self):
        """Test step execution."""
        def fn(a, b):
            return {"sum": a + b}

        step = Step(name="test-step", func=fn)
        # execute takes inputs dict
        result = step.execute({"a": 1, "b": 2})

        assert result.status == StepStatus.COMPLETED
        assert result.output["sum"] == 3

    def test_step_retry(self):
        """Test step retry on failure."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return {"x": 1}

        step = Step(name="test-step", func=fn, retries=3)
        result = step.execute({})

        assert result.status == StepStatus.COMPLETED
        assert call_count == 3


class TestDAG:
    """Test DAG class."""

    def test_dag_creation(self):
        """Test DAG creation."""
        dag = DAG()
        dag.add_node("a")
        # Add b depending on a
        dag.add_node("b", dependencies=["a"])

        assert "a" in dag
        assert "b" in dag

    def test_topological_sort(self):
        """Test topological sort."""
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b", dependencies=["a"])
        dag.add_node("c", dependencies=["b"])

        order = dag.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_cycle_detection(self):
        """Test cycle detection."""
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b", dependencies=["a"])
        # Create cycle
        # We need to manually manipulate edges or use add_node if re-adding logic was supported.
        # DAG.add_node overwrites? No, it just sets.
        # But we want to add edge b->a. b already depends on a.
        # Let's clean up:
        
        dag2 = DAG()
        dag2.add_node("a", dependencies=["b"])
        dag2.add_node("b", dependencies=["a"])

        with pytest.raises(ValueError, match="cycle"):
            dag2.topological_sort()

    def test_parallel_levels(self):
        """Test parallel execution levels."""
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c", dependencies=["a", "b"])
        dag.add_node("d", dependencies=["c"])

        levels = dag.get_parallel_levels()
        assert len(levels) == 3
        # levels[0] should be a and b (order independent, but sorted in output)
        assert set(levels[0]) == {"a", "b"}
        assert levels[1] == ["c"]
        assert levels[2] == ["d"]

