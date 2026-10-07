from functools import partial
from typing import Any, get_type_hints

from gradio.utils import get_function_params

from manual_graphrag.audit import (
    bind_gradio_callbacks_to_actor,
    current_actor,
)


class _ActorState:
    _id = 42


class _BlockFunction:
    def __init__(self):
        self.name = "sample_mutation"
        self.inputs = [object()]

        def callback(value):
            return current_actor(), value

        self.fn = callback


class _App:
    def __init__(self):
        self.config = {"dependencies": [{"id": 0, "inputs": [7]}]}
        self.fns = {0: _BlockFunction()}


def test_gradio_callbacks_receive_actor_in_isolated_context():
    app = _App()
    actor_state = _ActorState()

    bind_gradio_callbacks_to_actor(app, actor_state)

    assert app.fns[0].inputs[-1] is actor_state
    assert app.config["dependencies"][0]["inputs"] == [7, 42]
    assert app.fns[0].fn("payload", "Zhao") == ("Zhao", "payload")
    assert current_actor() is None


def test_partial_gradio_callback_keeps_annotation_module_for_api_schema():
    def callback(_action: str, value: Any) -> Any:
        return value

    app = _App()
    app.fns[0].fn = partial(callback, "remove")

    bind_gradio_callbacks_to_actor(app, _ActorState())

    wrapped = app.fns[0].fn
    assert wrapped.__module__ == callback.__module__
    assert get_type_hints(wrapped) == {
        "_action": str, "value": Any, "return": Any,
    }
    assert get_function_params(wrapped)
