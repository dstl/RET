"""Utility Methods for Creating Agents."""

# flake8: noqa: E501,W505

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, TypeVar, Union

from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.agents.airagent import AirAgent
from mesa_ret.agents.airdefenceagent import AirDefenceAgent
from mesa_ret.agents.groupagent import GroupAgent
from mesa_ret.agents.infantryagent import InfantryAgent
from mesa_ret.agents.protectedassetagent import ProtectedAssetAgent
from mesa_ret.agents.sensorfusionagent import SensorFusionAgent
from mesa_ret.behaviours.communicate import CommunicateMissionMessageBehaviour
from mesa_ret.space.heightband import HeightBand
from mesa_ret.weapons.weapon import Weapon

if TYPE_CHECKING:
    from typing import Any

    from mesa_ret.agents.affiliation import Affiliation
    from mesa_ret.behaviours import Behaviour
    from mesa_ret.behaviours.behaviourpool import ListAdder
    from mesa_ret.behaviours.communicate import (
        CommunicateOrdersBehaviour,
        CommunicateWorldviewBehaviour,
    )
    from mesa_ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
    from mesa_ret.behaviours.disablecommunication import DisableCommunicationBehaviour
    from mesa_ret.behaviours.fire import FireBehaviour
    from mesa_ret.behaviours.hide import HideBehaviour
    from mesa_ret.behaviours.move import MoveBehaviour
    from mesa_ret.behaviours.sense import SenseBehaviour
    from mesa_ret.behaviours.wait import WaitBehaviour
    from mesa_ret.communication.communicationreceiver import CommunicationReceiver
    from mesa_ret.model import RetModel
    from mesa_ret.orders.order import Order
    from mesa_ret.sensing.perceivedworld import PerceivedAgentFilter
    from mesa_ret.sensing.sensor import Sensor
    from mesa_ret.space.clutter.countermeasure import Countermeasure
    from mesa_ret.space.culture import Culture
    from mesa_ret.template import Template
    from mesa_ret.types import Coordinate2dOr3d

T = TypeVar("T")
SingleOrListOptional = Union[Optional[T], list[Optional[T]]]
OptionalSingleOrList = Optional[SingleOrListOptional[T]]


def create_agents(  # flake8: noqa: C501
    number: int,
    model: RetModel,
    pos: SingleOrListOptional[Coordinate2dOr3d],
    name: SingleOrListOptional[str],
    affiliation: SingleOrListOptional[Affiliation],
    agent_type: SingleOrListOptional[AgentType],
    critical_dimension: SingleOrListOptional[float] = None,
    reflectivity: SingleOrListOptional[float] = None,
    temperature: SingleOrListOptional[float] = None,
    temperature_std_dev: SingleOrListOptional[float] = None,
    height_bands: SingleOrListOptional[list[HeightBand]] = None,
    icon_path: OptionalSingleOrList[str] = None,
    killed_icon_path: OptionalSingleOrList[str] = None,
    orders: OptionalSingleOrList[list[Template[Order]]] = None,
    behaviour_adder: OptionalSingleOrList[type[ListAdder]] = None,
    move_behaviour: OptionalSingleOrList[MoveBehaviour] = None,
    wait_behaviour: OptionalSingleOrList[WaitBehaviour] = None,
    hide_behaviour: OptionalSingleOrList[HideBehaviour] = None,
    sense_behaviour: OptionalSingleOrList[SenseBehaviour] = None,
    sensors: OptionalSingleOrList[list[Template[Sensor]]] = None,
    communicate_orders_behaviour: OptionalSingleOrList[CommunicateOrdersBehaviour] = None,
    communicate_worldview_behaviour: OptionalSingleOrList[CommunicateWorldviewBehaviour] = None,
    communicate_mission_message_behaviour: OptionalSingleOrList[
        CommunicateMissionMessageBehaviour
    ] = None,
    fire_behaviour: OptionalSingleOrList[FireBehaviour] = None,
    disable_communication_behaviour: OptionalSingleOrList[DisableCommunicationBehaviour] = None,
    communication_receiver: OptionalSingleOrList[CommunicationReceiver] = None,
    refresh_technique: OptionalSingleOrList[PerceivedAgentFilter] = None,
    deploy_countermeasure_behaviour: OptionalSingleOrList[DeployCountermeasureBehaviour] = None,
    countermeasures: OptionalSingleOrList[Template[Countermeasure]] = None,
    weapons: OptionalSingleOrList[Weapon] = None,
    culture_speed_modifiers: OptionalSingleOrList[dict[Culture, float]] = None,
) -> list[RetAgent]:
    """Create and place multiple agents.

    Use the same parameters, or different parameters passed in as a list.

    If a list is passed in for any parameter its length must be equal to the number of
    agents being created.

    Args:
        number (int): Number of agents to be created.
        model (RetModel): The model the agents will be placed in.
        pos (SingleOrList[Coordinate2dOr3d]): The position for each agent.
        name (SingleOrList[str]): The name for each agent.
        affiliation (SingleOrList[Affiliation]): The affiliation for each agent.
        agent_type (SingleOrList[AgentType]): The type of each agent.
        critical_dimension (SingleOrListOptional[float]): Critical dimension of each agent.
        reflectivity (SingleOrListOptional[float]): Reflectivity of each agent.
        temperature (SingleOrListOptional[float]): Temperature of each agent.
        temperature_std_dev (SingleOrListOptional[float]): Temperature standard deviation of each agent.
        height_bands (SingleOrListOptional[list[HeightBand]]): Height bands for air agents.
        icon_path (OptionalSingleOrList[str]): path to the agent's icon.
        killed_icon_path (OptionalSingleOrList[str]): path to the agent's icon when killed.
        orders (OptionalSingleOrList[list[Template[Order]]]): The initial set of orders given to each agent. Defaults to None.
        behaviour_adder (OptionalSingleOrList[type[ListAdder]]): The behaviour adding methodology. Defaults to None.
        move_behaviour (OptionalSingleOrList[MoveBehaviour]): The behaviour of each agent when moving. Defaults to None.
        wait_behaviour (OptionalSingleOrList[WaitBehaviour]): The behaviour of each agent when waiting. Defaults to None.
        hide_behaviour (OptionalSingleOrList[WaitBehaviour]): The behaviour of each agent when hiding. Defaults to None.
        sense_behaviour (OptionalSingleOrList[SenseBehaviour]): The behaviour of each agent when sensing. Defaults to None.
        sensors (OptionalSingleOrList[list[Template[Sensor]]]): The sensor templates defining the sensors belonging to each agent. Defaults to None.
        communicate_orders_behaviour (OptionalSingleOrList[ CommunicateOrdersBehaviour]): The behaviour of each agent when communicating orders. Defaults to None.
        communicate_worldview_behaviour (OptionalSingleOrList[ CommunicateWorldviewBehaviour]): The behaviour of each agent when communicating World View. Defaults to None.
        communicate_mission_message_behaviour (OptionalSingleOrList[ CommunicateMissionMessageBehaviour]): The behaviour of each agent when communicating Mission messages. Defaults to None.
        fire_behaviour (OptionalSingleOrList[FireBehaviour]): The behaviour of each agent when firing. Defaults to None.
        disable_communication_behaviour (OptionalSingleOrList[ DisableCommunicationBehaviour]): The behaviour of each agent when disabling communications. Defaults to None.
        communication_receiver (OptionalSingleOrList[CommunicationReceiver]): Each agent's communication receiver. Defaults to None.
        refresh_technique (OptionalSingleOrList[PerceivedAgentFilter]): Methodology for each agent refreshing the perceived world. Defaults to None.
        deploy_countermeasure_behaviour (OptionalSingleOrList[ DeployCountermeasureBehaviour]): The behaviour of each agent when deploying a countermeasure. Defaults to None.
        countermeasures (OptionalSingleOrList[Template[Countermeasure]]): The templates of the countermeasures the agent has available. Defaults to None.
        weapons (OptionalSingleOrList[Weapon]): The weapon the agents have available. Defaults to None.
        culture_speed_modifiers (OptionalSingleOrList[dict[Culture,float]]): The speed modifiers for the agent for each culture in the space. Defaults to None.

    Returns:
        agent_list (list[RetAgent]): A list of the agents created.
    """
    _pos: list[Coordinate2dOr3d] = create_list(number, pos)  # type: ignore
    _name: list[str] = create_list(number, name)  # type: ignore
    _affiliation: list[Affiliation] = create_list(number, affiliation)  # type: ignore
    _agent_type: list[AgentType] = create_list(number, agent_type)  # type: ignore
    _critical_dimension: list[float] = create_list(number, critical_dimension)  # type: ignore
    _reflectivity: list[float] = create_list(number, reflectivity)  # type: ignore
    _temperature: list[float] = create_list(number, temperature)  # type: ignore
    _temperature_std_dev: list[float] = create_list(number, temperature_std_dev)  # type: ignore
    _height_bands: list[list[HeightBand]] = create_list_of_lists(number, height_bands)  # type: ignore
    _icon_path: list[str] = create_list(number, icon_path)  # type: ignore
    _killed_icon_path: list[str] = create_list(number, killed_icon_path)  # type: ignore
    _orders: list[list[Template[Order]]] = create_list_of_lists(number, orders)  # type: ignore
    _behaviour_adder: list[type[ListAdder]] = create_list(number, behaviour_adder)  # type: ignore
    _move_behaviour: list[MoveBehaviour] = create_list(number, move_behaviour)  # type: ignore
    _wait_behaviour: list[WaitBehaviour] = create_list(number, wait_behaviour)  # type: ignore
    _hide_behaviour: list[HideBehaviour] = create_list(number, hide_behaviour)  # type: ignore
    _sense_behaviour: list[SenseBehaviour] = create_list(number, sense_behaviour)  # type: ignore
    _sensors: list[list[Template[Sensor]]] = create_list_of_lists(number, sensors)  # type: ignore
    _communicate_orders_behaviour: list[CommunicateOrdersBehaviour] = create_list(
        number, communicate_orders_behaviour
    )  # type: ignore
    _communicate_worldview_behaviour: list[CommunicateWorldviewBehaviour] = create_list(
        number, communicate_worldview_behaviour
    )  # type: ignore
    _communicate_mission_message_behaviour: list[CommunicateMissionMessageBehaviour] = create_list(
        number, communicate_mission_message_behaviour
    )  # type: ignore
    _fire_behaviour: list[FireBehaviour] = create_list(number, fire_behaviour)  # type: ignore
    _disable_communication_behaviour: list[DisableCommunicationBehaviour] = create_list(
        number, disable_communication_behaviour
    )  # type: ignore
    _communication_receiver: list[CommunicationReceiver] = create_list(
        number, communication_receiver
    )  # type: ignore
    _refresh_technique: list[PerceivedAgentFilter] = create_list(
        number, refresh_technique
    )  # type: ignore
    _deploy_countermeasure_behaviour: list[DeployCountermeasureBehaviour] = create_list(
        number, deploy_countermeasure_behaviour
    )  # type: ignore
    _countermeasures: list[list[Template[Countermeasure]]] = create_list_of_lists(
        number, countermeasures  # type: ignore
    )  # type: ignore

    _weapons: list[list[Weapon]] = create_list_of_lists(number, weapons)  # type: ignore
    _culture_speed_modifiers: list[dict[Culture, float]] = create_list(number, culture_speed_modifiers)  # type: ignore

    parameter_lists: dict[str, list[Any]] = {
        "pos": _pos,
        "name": _name,
        "affiliation": _affiliation,
        "critical_dimension": _critical_dimension,
        "reflectivity": _reflectivity,
        "temperature": _temperature,
        "temperature_std_dev": _temperature_std_dev,
        "height_bands": _height_bands,
        "icon_path": _icon_path,
        "killed_icon_path": _killed_icon_path,
        "orders": _orders,
        "sensors": _sensors,
        "communication_receiver": _communication_receiver,
        "refresh_technique": _refresh_technique,
        "countermeasures": _countermeasures,
        "weapons": _weapons,
        "behaviour_adder": _behaviour_adder,
        "countermeasures": _countermeasures,
        "height_bands": _height_bands,
        "culture_speed_modifiers": _culture_speed_modifiers,
    }

    agent_list = []
    kwargs: dict[str, Any] = {}
    for a in range(0, number):
        optional_behaviours: list[Optional[Behaviour]] = [
            _move_behaviour[a],
            _wait_behaviour[a],
            _hide_behaviour[a],
            _sense_behaviour[a],
            _communicate_orders_behaviour[a],
            _communicate_worldview_behaviour[a],
            _communicate_mission_message_behaviour[a],
            _fire_behaviour[a],
            _disable_communication_behaviour[a],
            _deploy_countermeasure_behaviour[a],
        ]
        behaviours: list[Behaviour] = [
            behaviour for behaviour in optional_behaviours if behaviour is not None
        ]
        if _agent_type[a] is AgentType.AIR:
            invalid_arguments = ["culture_speed_modifiers"]
            kwargs = update_kwargs(model, parameter_lists, invalid_arguments, a, "AirAgent")

            kwargs["behaviours"] = behaviours
            agent = AirAgent(**kwargs)

        elif _agent_type[a] is AgentType.AIR_DEFENCE:
            invalid_arguments = ["countermeasures", "height_bands"]
            kwargs = update_kwargs(model, parameter_lists, invalid_arguments, a, "AirDefenceAgent")

            kwargs["behaviours"] = behaviours
            agent = AirDefenceAgent(**kwargs)

        elif _agent_type[a] is AgentType.GROUP:
            invalid_arguments = [
                "pos",
                "temperature",
                "temperature_std_dev",
                "reflectivity",
                "sensors",
                "countermeasures",
                "height_bands",
                "communication_receiver",
                "culture_speed_modifiers",
                "refresh_technique",
                "weapons",
            ]
            kwargs = update_kwargs(model, parameter_lists, invalid_arguments, a, "GroupAgent")

            kwargs["communicate_orders_behaviour"] = _communicate_orders_behaviour[a]
            agent = GroupAgent(**kwargs)

        elif _agent_type[a] is AgentType.INFANTRY:
            invalid_arguments = ["countermeasures", "height_bands"]
            kwargs = update_kwargs(model, parameter_lists, invalid_arguments, a, "InfantryAgent")

            kwargs["behaviours"] = behaviours
            agent = InfantryAgent(**kwargs)

        elif _agent_type[a] is AgentType.PROTECTED_ASSET:
            invalid_arguments = [
                "sensors",
                "countermeasures",
                "height_bands",
                "communication_receiver",
                "culture_speed_modifiers",
                "refresh_technique",
                "weapons",
                "behaviour_adder",
                "orders",
            ]
            kwargs = update_kwargs(
                model, parameter_lists, invalid_arguments, a, "ProtectedAssetAgent"
            )

            agent = ProtectedAssetAgent(**kwargs)

        elif _agent_type[a] is AgentType.SENSOR_FUSION:
            invalid_arguments = [
                "sensors",
                "countermeasures",
                "height_bands",
                "communication_receiver",
                "weapons",
                "orders",
            ]
            kwargs = update_kwargs(
                model, parameter_lists, invalid_arguments, a, "SensorFusionAgent"
            )

            kwargs["behaviours"] = behaviours
            agent = SensorFusionAgent(**kwargs)

        else:
            invalid_arguments = ["culture_speed_modifiers"]
            kwargs = update_kwargs(model, parameter_lists, invalid_arguments, a, "RetAgent")
            kwargs["behaviours"] = behaviours
            if parameter_lists["critical_dimension"][a] is None:
                kwargs["critical_dimension"] = None
            agent = RetAgent(**kwargs)

        agent_list.append(agent)

    return agent_list


def create_list(number: int, to_check: SingleOrListOptional[T]) -> list[T]:
    """Create list from a SingleOrList.

    If length is equal to the number, returns the list.

    If input not iterable, makes a list of length number with each entry the same then
    returns that list.

    Args:
        number (int): The number of entries to check that the list should have.
        to_check (SingleOrList[T]): The list who's length will be checked against
            the number.

    Raises:
        IndexError: Raises an exception if length of list does not equal number.

    Returns:
        list_to_return (list[T]): The original input parameter if it was a list.
            Otherwise a list of length number with all entries being the to_check input
            parameter.
    """
    if isinstance(to_check, list):
        if len(to_check) == number:
            list_to_return: list[T] = to_check  # type: ignore
        else:
            raise IndexError("List length incompatible with number of agents. No agents created.")
    else:
        list_to_return = [to_check] * number  # type: ignore

    return list_to_return


def create_list_of_lists(number: int, to_check: SingleOrListOptional[list[T]]) -> list[list[T]]:
    """Create list from SingleOrList[List].

    If length is equal to the number, returns the list.

    If input not iterable, makes a list of length number with each entry the same then
    returns that list.

    Args:
        number (int): The number of entries to check that the list should have.
        to_check (SingleOrListOptional[list[T]]): The list who's length will be checked against
            the number.

    Raises:
        IndexError: Raises an exception if length of list does not equal number.
        TypeError: Parameter must be None, a list of single objects, or a list of lists. Cannot have
            lists and single objects in the same list.

    Returns:
        list_to_return (list[T]): The original input parameter if it was a list of
            lists. Otherwise a list of length number with all entries being the
            to_check input parameter.
    """
    _none_counter = 0
    _list_counter = 0
    list_to_return: list[list[T]] = []
    #  Check if the parameter passed in is a list
    if isinstance(to_check, list):
        # If yes, check that all elements are either None, or a list
        for element in to_check:
            if element is None:
                _none_counter += 1
            elif isinstance(element, list):
                _list_counter += 1

        if (_none_counter + _list_counter) == len(to_check):
            if len(to_check) == number:
                list_to_return = to_check  # type: ignore
            else:
                raise IndexError(
                    "List length incompatible with number of agents. No agents created."
                )

        elif _list_counter == 0:
            # If only None or non-lists
            list_to_return = [to_check] * number  # type: ignore
        else:
            raise TypeError("Cannot have lists and single objects in the same list.")

    elif to_check is None:
        list_to_return = [to_check] * number  # type: ignore

    else:
        raise TypeError("Parameter must be None, a list of single objects, or a list of lists.")

    return list_to_return


def update_kwargs(
    model: RetModel,
    parameter_lists: dict[str, list[Any]],
    invalid_arguments: list[str],
    agent_number: int,
    class_type: str,
) -> dict[str, Any]:
    """Update and generate the kwargs for creating a new agent.

    Args:
        model (RetModel): The model to add the agent to.
        parameter_lists (dict[str, list[Any]]): A dictionary of all the parameter lists containing
            the necessary arguments to create the new agent.
        invalid_arguments (list[str]): Parameter lists to skip which are not relevant for this agent
            type.
        agent_number (int): The index to use in the parameter lists to find relevant parameters.
        class_type (str): The name for the agent type (to go in any warning messages).

    Returns:
        dict[str, Any]: key word arguments for generating an agent.
    """
    return_kwargs = {"model": model}

    for arg_name, arg_list in parameter_lists.items():
        if arg_list[agent_number] is not None:
            if arg_name in invalid_arguments:
                agent_name = parameter_lists["name"][agent_number]
                warnings.warn(
                    f"'{arg_name}' has been defined for agent {agent_name} "
                    + f"of type {class_type}, but is not a valid input for this agent class."
                )
            else:
                return_kwargs[arg_name] = arg_list[agent_number]

    return return_kwargs
