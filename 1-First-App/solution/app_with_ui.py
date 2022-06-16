from lightning import LightningApp, LightningFlow, LightningWork
from lightning.app.components.python.popen import PopenPythonScript
from lightning.app.frontend import StreamlitFrontend


class Main(LightningFlow):
    def __init__(self):
        super().__init__()
        self.ui = UI()
        self.script_runner = PopenPythonScript(script_path="project/train_cifar.py")

    def run(self):
        if self.ui.run_script:
            self.script_runner.run()

    def configure_layout(self):
        return {"name": "Training", "content": self.ui}


class UI(LightningFlow):
    def __init__(self):
        super().__init__()
        self.run_script = False

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render)


def render(app_state):
    import streamlit

    app_state.run_script = streamlit.button("Run training")


app = LightningApp(Main())
