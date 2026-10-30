"""Cross-package contracts required by the planar-search runtime."""

from pathlib import Path
import threading
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import yaml

from s622_dual_assembly.planar_search_node import PlanarSearchNode


SRC = Path(__file__).parents[2]


def dependencies(package):
    root = ET.parse(SRC / package / "package.xml").getroot()
    return {
        element.text for element in root
        if element.tag in {"depend", "exec_depend", "build_depend"}
    }


def test_search_axes_force_sign_and_runtime_topics_are_explicit():
    path = SRC / "s622_dual_assembly" / "config" / "planar_search.yaml"
    parameters = yaml.safe_load(path.read_text(encoding="utf-8"))[
        "/planar_search_node"
    ]["ros__parameters"]
    assert parameters["command_frame"] == "right_grasp_frame"
    assert parameters["desired_contact_force"] > 0.0
    assert parameters["velocity_topic"] == (
        "/right_admittance/nominal_velocity_command"
    )
    assert parameters["wrench_topic"] == (
        "/right_ft_sensor_processing/wrench_filtered"
    )
    assert parameters["approach_speed"] > 0.0
    assert parameters["max_approach_distance"] > 0.0


def test_action_and_dynamic_admittance_contract_are_installed():
    action = (
        SRC / "s622_dual_interfaces" / "action" / "PlanarSearch.action"
    ).read_text(encoding="utf-8")
    service = (
        SRC / "s622_dual_interfaces" / "srv" / "ConfigureAdmittance.srv"
    ).read_text(encoding="utf-8")
    controller = (
        SRC / "s622_dual_admittance" / "nodes" / "admittance_node.cpp"
    ).read_text(encoding="utf-8")
    assert "bool execute" in action
    assert "bool require_velocity_command" in service
    assert "bool bypass_admittance" in service
    assert '"nominal_velocity_command"' in controller
    assert 'publish_status("STALE_COMMAND"' in controller


def test_runtime_dependencies_include_assembly_boundaries():
    assembly = dependencies("s622_dual_assembly")
    task = dependencies("s622_dual_task")
    bringup = dependencies("s622_dual_bringup")
    assert {"rclpy", "s622_dual_interfaces", "tf2_ros"} <= assembly
    assert "s622_dual_assembly" in task
    assert "s622_dual_assembly" in bringup


def test_search_only_requires_configure_and_coordinated_enable_services():
    node = PlanarSearchNode.__new__(PlanarSearchNode)
    node._configure_client = NS(service_is_ready=lambda: True)
    node._enable_client = NS(service_is_ready=lambda: True)
    assert node._interfaces_ready()
    node._enable_client = NS(service_is_ready=lambda: False)
    assert not node._interfaces_ready()


def test_search_confirms_disable_even_when_enable_does_not_succeed():
    node = PlanarSearchNode.__new__(PlanarSearchNode)
    calls = []
    errors = []
    node._core = NS(start=lambda: NS(desired_force_y=0.0, phase=NS(value='ESTABLISH_CONTACT')), radius=0.0)
    node._config = NS(desired_contact_force=10.0)
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._wait_for_interfaces = lambda: True
    node._configure_admittance = lambda force: calls.append('configure')
    node._enable_client = object()
    node._publish_state = lambda *args: None
    node._publish_velocity = lambda command: calls.append('zero')
    node.get_logger = lambda: NS(error=errors.append)

    def call(client, request):
        assert client is node._enable_client
        calls.append(request.data)
        if request.data:
            raise RuntimeError('enable response timed out')
    node._call = call
    goal = NS(is_active=True, abort=lambda: calls.append('abort'))
    result = node._execute(goal)
    assert not result.success
    assert result.reason == 'enable response timed out'
    assert calls == ['configure', True, 'abort', 'zero', False]
    assert not node._goal_active
