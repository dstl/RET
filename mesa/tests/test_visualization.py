import os
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Union
from unittest import TestCase

from mesa import Agent
from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import ContinuousSpace, Grid, HexGrid
from mesa.time import BaseScheduler, SimultaneousActivation
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import (
    BarChartModule,
    CanvasContinuous,
    CanvasGrid,
    CanvasHexGrid,
    ChartModule,
    PieChartModule,
    TextElement,
)
from mesa.visualization.UserParam import UserSettableParameter

from tests.test_batchrunner import MockAgent


class MockModel(Model):
    """Test model for testing"""

    def __init__(self, width, height, key1=103, key2=104):

        self.width = width
        self.height = height
        self.key1 = (key1,)
        self.key2 = key2
        self.schedule = SimultaneousActivation(self)
        self.grid = Grid(width, height, torus=True)

        for (c, x, y) in self.grid.coord_iter():
            a = MockAgent(x + y * 100, self, x * y * 3)
            self.grid.place_agent(a, (x, y))
            self.schedule.add(a)

    def step(self):
        self.schedule.step()


class TestModularServer(TestCase):
    """Test server for testing"""

    def portrayal(self, cell):
        return {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Layer": 0,
            "x": 0,
            "y": 0,
            "Color": "black",
        }

    def setUp(self):

        self.user_params = {
            "width": 1,
            "height": 1,
            "key1": UserSettableParameter("number", "Test Parameter", 101),
            "key2": UserSettableParameter("slider", "Test Parameter", 200, 0, 300, 10),
        }

        self.viz_elements = [
            CanvasGrid(self.portrayal, 10, 10, 20, 20),
            TextElement(),
            # ChartModule([{"Label": "Wolves", "Color": "#AA0000"},  # Todo - test chart module
            #              {"Label": "Sheep", "Color": "#666666"}])
        ]

        self.server = ModularServer(
            MockModel, self.viz_elements, "Test Model", model_params=self.user_params
        )

    def test_canvas_render_model_state(self):

        test_portrayal = self.portrayal(None)
        test_grid_state = defaultdict(list)
        test_grid_state[test_portrayal["Layer"]].append(test_portrayal)

        state = self.server.render_model()
        assert state[0] == test_grid_state

    def test_text_render_model_state(self):
        state = self.server.render_model()
        assert state[1] == "<b>VisualizationElement goes here</b>."

    def test_user_params(self):
        print(self.server.user_params)
        assert self.server.user_params == {
            "key1": UserSettableParameter("number", "Test Parameter", 101).json,
            "key2": UserSettableParameter(
                "slider", "Test Parameter", 200, 0, 300, 10
            ).json,
        }


class MockAgent2(Agent):
    """Minimalistic agent implementation for testing purposes."""

    def __init__(self, unique_id, model, value):
        super().__init__(unique_id, model)
        self.value = value


class MockModel2(Model):
    """Mock model base class for testing"""

    def __init__(self, width, height):

        self.schedule = BaseScheduler(self)
        self.space = self.get_space(width, height)
        self.grid = self.space

        pos_start = (10, 10)
        agents = [{"number": 1, "value": 1}, {"number": 2, "value": 2}]

        model_reporters = {}
        agent_reporters = {
            "value": lambda a: a.value,
        }
        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        for a in agents:
            agent = MockAgent2(a["number"], self, a["value"])
            self.space.place_agent(agent, pos_start)
            self.schedule.add(agent)

    @abstractmethod
    def get_space(self, width, height) -> Union[Grid, HexGrid, ContinuousSpace]:
        pass

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


class MockContinuousModel(MockModel2):
    """Mock continuous model for testing."""

    def get_space(self, width, height) -> ContinuousSpace:
        return ContinuousSpace(width, height, False)


class MockHexGridModel(MockModel2):
    """Mock hex grid model for testing."""

    def get_space(self, width, height) -> HexGrid:
        return HexGrid(width, height, False)


class MockGridModel(MockModel2):
    """Mock grid model for testing."""

    def get_space(self, width, height) -> Grid:
        return Grid(width, height, False)


class VisualizationTestingBase:
    """Base class to test visualisations."""

    @abstractmethod
    def get_canvas(self, *args, **kwargs) -> Union[CanvasGrid, CanvasHexGrid]:
        # CanvasGrid and CanvasHexGrid options are returned because there is no shared
        # base class.
        pass

    @abstractmethod
    def get_model(self, *args, **kwargs) -> Model:
        pass

    @abstractmethod
    def get_canvas_module_name(self) -> str:
        return ""

    def unit_portrayal(self, agent: MockAgent2):
        portrayal = {"Layer": 0}
        return portrayal

    def test_initialise(self):
        canvas = self.get_canvas(None, 100, 100)
        assert canvas.canvas_width == canvas.canvas_height == 500
        assert canvas.grid_height == canvas.grid_width == 100
        assert canvas.canvas_background_path == ""

    def test_with_background(self):
        canvas_background_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_background.png")
        )
        canvas = self.get_canvas(
            None, 100, 100, canvas_background_file=canvas_background_path
        )
        assert (
            canvas.js_code
            == f"elements.push(new {self.get_canvas_module_name()}(500, 500, 100, 100, '/assets/background_image.png'));"
        )
        assert canvas.canvas_background_path == "/assets/background_image.png"

    def test_render(self):
        canvas = self.get_canvas(self.unit_portrayal, 100, 100)
        assert canvas.render(self.get_model(100, 100)) == defaultdict(
            list, {0: [{"Layer": 0, "x": 10, "y": 10}]}
        )


class TestGridVisualization(TestCase, VisualizationTestingBase):
    """Tests for the grid visualization."""

    def get_canvas(self, *args, **kwargs) -> CanvasGrid:
        return CanvasGrid(*args, **kwargs)

    def get_model(self, *args, **kwargs) -> MockGridModel:
        return MockGridModel(*args, **kwargs)

    def get_canvas_module_name(self) -> str:
        return "CanvasModule"


class TestHexGridVisualization(TestCase, VisualizationTestingBase):
    """Tests for the hex grid visualization."""

    def get_canvas(self, *args, **kwargs) -> CanvasHexGrid:
        return CanvasHexGrid(*args, **kwargs)

    def get_model(self, *args, **kwargs) -> MockHexGridModel:
        return MockHexGridModel(*args, **kwargs)

    def get_canvas_module_name(self) -> str:
        return "CanvasHexModule"


class TestContinuousVisualization(TestCase):
    """Tests for the continuous visualization."""

    def unit_portrayal(self, agent: MockAgent2):
        portrayal = {"Value": agent.value, "Layer": 0}
        return portrayal

    def test_initialise(self):
        canvas = CanvasContinuous(None, 100, 100)
        assert canvas.canvas_width == canvas.canvas_height == 100
        assert canvas.canvas_background_path == ""

    def test_with_background(self):
        canvas_background_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_background.png")
        )
        canvas = CanvasContinuous(self.unit_portrayal, 100, 100, canvas_background_file)
        assert canvas.canvas_background_path == "/assets/background_image.png"
        assert (
            canvas.js_code
            == "elements.push(new CanvasContinuousModule(100, 100, '/assets/background_image.png'));"
        )

    def test_invalid_background(self):
        canvas_background_file = "Test"
        with self.assertRaises(FileNotFoundError) as cm:
            CanvasContinuous(self.unit_portrayal, 100, 100, canvas_background_file)
        assert cm.exception.errno == 2
        assert cm.exception.strerror == "No such file or directory"

    def test_background_image_stretching(self):
        canvas_background_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_background.png")
        )
        with warnings.catch_warnings(record=True) as w:
            CanvasContinuous(self.unit_portrayal, 100, 1, canvas_background_file)
        assert len(w) == 1
        assert isinstance(w[0], warnings.WarningMessage)
        assert (
            "image aspect ratio differs from the space aspect ratio by more than 5%,"
            in str(w[0].message)
        )

    def test_render(self):
        canvas = CanvasContinuous(self.unit_portrayal, 100, 100)
        assert canvas.render(MockContinuousModel(100, 100)) == defaultdict(
            list,
            {
                0: [
                    {"Value": 1, "Layer": 0, "x": 0.1, "y": 0.1},
                    {"Value": 2, "Layer": 0, "x": 0.1, "y": 0.1},
                ]
            },
        )


class TestBarChartVisualization(TestCase):
    """Tests for the bar chart visualization."""

    def test_BarChartModule_initialise(self):
        chart = BarChartModule(
            [{"Label": "Field name", "Color": "#ff0000"}],
            canvas_height=100,
            canvas_width=100,
        )
        assert chart.fields == [{"Label": "Field name", "Color": "#ff0000"}]

    def test_BarChartModule_canvas(self):
        chart = BarChartModule(
            [{"Label": "Field name", "Color": "#ff0000"}],
            canvas_height=100,
            canvas_width=100,
        )
        assert chart.canvas_height == chart.canvas_width == 100

    def test_BarChartModule_render_method_no_scope(self):
        chart = BarChartModule(
            [{"Label": "Field name", "Color": "#ff0000"}], scope="Test"
        )
        with self.assertRaises(ValueError) as ve:
            chart.render(MockContinuousModel(100, 100))
        assert ve.exception.args[0] == "scope must be 'agent' or 'model'"

    def test_BarChartModule_render_method_agent_scope(self):
        model = MockContinuousModel(100, 100)
        chart = BarChartModule(
            [{"Label": "value", "Color": "#ff0000"}],
            scope="agent",
            data_collector_name="datacollector",
        )
        model.step()
        assert chart.render(model) == [{"value": 1}, {"value": 2}]

    def test_BarChartModule_render_method_model_scope(self):
        chart = BarChartModule(
            [{"Label": "Field name", "Color": "#ff0000"}], scope="model"
        )
        assert chart.render(MockContinuousModel(100, 100)) == [{"Field name": 0}]


class TestChartVisualization(TestCase):
    """Tests for the chart visualization."""

    def test_ChartModule_initialise(self):
        chart = ChartModule([{"Label": "Field name", "Color": "#ff0000"}])
        assert chart.series == [{"Label": "Field name", "Color": "#ff0000"}]

    def test_ChartModule_canvas(self):
        chart = ChartModule(
            [{"Label": "Field name", "Color": "#ff0000"}],
            canvas_height=100,
            canvas_width=100,
        )
        assert chart.canvas_height == chart.canvas_width == 100

    def test_ChartModule_render_method_model_scope(self):
        chart = ChartModule([{"Label": "Field name", "Color": "#ff0000"}])
        assert chart.render(MockContinuousModel(100, 100)) == [0]


class TestPieChartVisualization(TestCase):
    """Tests for the Pie chart visualization."""

    def test_PieChartModule_initialise(self):
        chart = PieChartModule([{"Label": "Field name", "Color": "#ff0000"}])
        assert chart.fields == [{"Label": "Field name", "Color": "#ff0000"}]

    def test_PieChartModule_canvas(self):
        chart = PieChartModule(
            [{"Label": "Field name", "Color": "#ff0000"}],
            canvas_height=100,
            canvas_width=100,
        )
        assert chart.canvas_height == chart.canvas_width == 100

    def test_PieChartModule_render_method_model_scope(self):
        chart = PieChartModule([{"Label": "Field name", "Color": "#ff0000"}])
        assert chart.render(MockContinuousModel(100, 100)) == [0]
