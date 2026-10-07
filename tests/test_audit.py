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
