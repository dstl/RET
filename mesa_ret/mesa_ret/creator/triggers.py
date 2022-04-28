"""Utility Methods for Creating Triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, TypeVar, Union

from mesa_ret.creator.agents import create_list
from mesa_ret.orders.triggers.immediate import ImmediateSensorFusionTrigger, ImmediateTrigger
from mesa_ret.orders.triggers.killed import AgentKilledTrigger, KilledAgentsAtPositionTrigger
from mesa_ret.orders.triggers.position import (
    AliveAgentsAtPositionTrigger,
    CrossedBoundaryTrigger,
    InAreaTrigger,
    MovedOutOfAreaTrigger,
    NotInAreaTrigger,
    PositionTrigger,
)
from mesa_ret.orders.triggers.time import TimeTrigger
from mesa_ret.orders.triggers.triggertype import TriggerType
from mesa_ret.orders.triggers.weapon import (
    AgentFiredWeaponTrigger,
    WeaponFiredNearAgentTrigger,
    WeaponFiredNearLocationTrigger,
)

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.orders.order import Trigger
    from mesa_ret.space.feature import Area, Boundary
    from mesa_ret.types import Coordinate2dOr3d

T = TypeVar("T")

SingleOrListOptional = Union[Optional[T], list[Optional[T]]]
OptionalSingleOrList = Optional[SingleOrListOptional[T]]


@dataclass
class TriggerConfig:
    """RET Trigger creation configuration."""

    sticky_list: list[bool] = field(default_factory=list)
    log_list: list[bool] = field(default_factory=list)
    invert_list: list[bool] = field(default_factory=list)
    position_list: list[Coordinate2dOr3d] = field(default_factory=list)
    agent_list: list[RetAgent] = field(default_factory=list)
    tolerance_list: list[float] = field(default_factory=list)
    area_list: list[Area] = field(default_factory=list)
    boundary_list: list[Boundary] = field(default_factory=list)
    time_list: list[datetime] = field(default_factory=list)
    weapon_name_list: list[str] = field(default_factory=list)


def _create_alive_agents_at_position_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an alive agents at position trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.position_list[trigger_counter],
            trigger_config_lists.tolerance_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        AliveAgentsAtPositionTrigger(
            position=trigger_config_lists.position_list[trigger_counter],
            tolerance=trigger_config_lists.tolerance_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
            invert=optional_args[2],
        )
    )


def _create_agent_killed_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent killed trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        AgentKilledTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
            invert=optional_args[2],
        )
    )


def _create_agent_at_position_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent at position trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.position_list[trigger_counter],
            trigger_config_lists.agent_list[trigger_counter],
            trigger_config_lists.tolerance_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        PositionTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            position=trigger_config_lists.position_list[trigger_counter],
            tolerance=trigger_config_lists.tolerance_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
            invert=optional_args[2],
        )
    )


def _create_agent_fired_weapon_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent fired weapon trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        AgentFiredWeaponTrigger(
            firer=trigger_config_lists.agent_list[trigger_counter],
            weapon_name=trigger_config_lists.weapon_name_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
        )
    )


def _create_weapon_fired_near_agent_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an weapon fired near agent trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
            trigger_config_lists.tolerance_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        WeaponFiredNearAgentTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            weapon_name=trigger_config_lists.weapon_name_list[trigger_counter],
            tolerance=trigger_config_lists.tolerance_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
        )
    )


def _create_weapon_fired_near_location_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an weapon fired near location trigger using a TriggerConfig and add to trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.position_list[trigger_counter],
            trigger_config_lists.tolerance_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        WeaponFiredNearLocationTrigger(
            location=trigger_config_lists.position_list[trigger_counter],
            weapon_name=trigger_config_lists.weapon_name_list[trigger_counter],
            tolerance=trigger_config_lists.tolerance_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
        )
    )


def _create_time_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create a time trigger using a trigger configuration and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none([trigger_config_lists.time_list[trigger_counter]])
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        TimeTrigger(
            time=trigger_config_lists.time_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
            invert=optional_args[2],
        )
    )


def _create_agent_moved_out_of_area_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent moved out of area trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
            trigger_config_lists.area_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        MovedOutOfAreaTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            area=trigger_config_lists.area_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
            invert=optional_args[2],
        )
    )


def _create_agent_crossed_boundary_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent crossed boundary trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
            trigger_config_lists.boundary_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        CrossedBoundaryTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            boundary=trigger_config_lists.boundary_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
            invert=optional_args[2],
        )
    )


def _create_agent_not_in_area_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent not in area trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
            trigger_config_lists.area_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        NotInAreaTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            area=trigger_config_lists.area_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
        )
    )


def _create_agent_in_area_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an agent in area trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none(
        [
            trigger_config_lists.agent_list[trigger_counter],
            trigger_config_lists.area_list[trigger_counter],
        ]
    )
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        InAreaTrigger(
            agent=trigger_config_lists.agent_list[trigger_counter],
            area=trigger_config_lists.area_list[trigger_counter],
            sticky=optional_args[0],
            log=optional_args[1],
        )
    )


def _create_killed_agents_at_position_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create a killed agents at position trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    _check_arg_not_none([trigger_config_lists.position_list[trigger_counter]])
    if trigger_config_lists.sticky_list[trigger_counter] is None:
        trigger_config_lists.sticky_list[trigger_counter] = True
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(
        KilledAgentsAtPositionTrigger(
            position=trigger_config_lists.position_list[trigger_counter],
            sticky=optional_args[1],
            log=optional_args[0],
            invert=optional_args[2],
        )
    )


def _create_immediate_sensor_fusion_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an immediate sensor fusion trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(ImmediateSensorFusionTrigger(log=optional_args[1]))


def _create_immediate_trigger(
    trigger_counter: int, trigger_config_lists: TriggerConfig, trigger_list: list[Trigger]
) -> None:
    """Create an immediate trigger using a TriggerConfig and add to the trigger list.

    Args:
        trigger_counter (int): Which trigger element to create. This indicates the element to use
            from the list of configuration settings.
        trigger_config_lists (TriggerConfig): The trigger configuration object holding all the lists
            of configuration settings.
        trigger_list (list[Trigger]): The list of created triggers to be added to.
    """
    optional_args = _check_optional_arg_not_none(
        trigger_config_lists.sticky_list[trigger_counter],
        trigger_config_lists.log_list[trigger_counter],
        trigger_config_lists.invert_list[trigger_counter],
    )
    trigger_list.append(ImmediateTrigger(log=optional_args[1]))


def _invalid_trigger_type(*args):
    """Raise an Error for unhandled Trigger type.

    Raises:
        TypeError: The trigger type was not recognised
    """
    raise TypeError("Selected Trigger Type does not exist.")


def _create_configuration_lists(
    number: int,
    sticky: SingleOrListOptional[bool],
    log: SingleOrListOptional[bool],
    invert: SingleOrListOptional[bool],
    position: OptionalSingleOrList[Coordinate2dOr3d],
    agent: OptionalSingleOrList[RetAgent],
    tolerance: OptionalSingleOrList[float],
    area: OptionalSingleOrList[Area],
    boundary: OptionalSingleOrList[Boundary],
    time: OptionalSingleOrList[datetime],
    weapon_name: OptionalSingleOrList[str],
) -> TriggerConfig:
    """Create a trigger configuration object holding the lists of required arguments.

    Args:
        number (int): The number of triggers to create.
        sticky (SingleOrListOptional[bool]): The sticky parameter for each trigger.
        log (SingleOrListOptional[bool]): The log parameter for each trigger.
        invert (SingleOrListOptional[bool]): The invert parameter for each trigger.
        position (OptionalSingleOrList[Coordinate2dOr3d]): The position parameter for each trigger.
            Required by KILLED_AGENTS_AT_POSITION and AGENT_AT_POSITION triggers.
        agent (OptionalSingleOrList[RetAgent]): The agent parameter for each trigger. Required by
            AGENT_AT_POSITION, AGENT_IN_AREA, AGENT_NOT_IN_AREA, AGENT_CROSSED_BOUNDARY, and
            AGENT_MOVED_OUT_OF_AREA triggers.
        tolerance (OptionalSingleOrList[float]): The tolerance parameter for each trigger. Required
            by AGENT_AT_POSITION trigger.
        area (OptionalSingleOrList[Area]): The area parameter for each trigger. Required by
            AGENT_IN_AREA, AGENT_NOT_IN_AREA, and AGENT_MOVED_OUT_OF_AREA triggers.
        boundary (OptionalSingleOrList[Boundary]): The boundary paramater for each trigger.
            Required by AGENT_CROSSED_BOUNDARY trigger.
        time (OptionalSingleOrList[datetime]): The time paramater for each trigger. Required by
            TIME trigger.
        weapon_name (OptionalSingleOrList[str]): The weapon name paramater for each trigger.
            Optionally required by AGENT_FIRED_WEAPON, WEAPON_FIRED_NEAR_LOCATION, and
            WEAPON_FIRED_NEAR_AGENT triggers.

    Returns:
        TriggerConfig: A trigger configuration object holding the trigger configuration
    """
    trigger_config_lists: TriggerConfig = TriggerConfig()

    trigger_config_lists.sticky_list = create_list(number, sticky)
    trigger_config_lists.log_list = create_list(number, log)
    trigger_config_lists.invert_list = create_list(number, invert)
    trigger_config_lists.position_list = create_list(number, position)
    trigger_config_lists.agent_list = create_list(number, agent)
    trigger_config_lists.tolerance_list = create_list(number, tolerance)
    trigger_config_lists.area_list = create_list(number, area)
    trigger_config_lists.boundary_list = create_list(number, boundary)
    trigger_config_lists.time_list = create_list(number, time)
    trigger_config_lists.weapon_name_list = create_list(number, weapon_name)

    return trigger_config_lists


def create_triggers(
    number: int,
    trigger_type: SingleOrListOptional[TriggerType],
    sticky: SingleOrListOptional[bool] = False,
    log: SingleOrListOptional[bool] = True,
    invert: SingleOrListOptional[bool] = False,
    position: OptionalSingleOrList[Coordinate2dOr3d] = None,
    agent: OptionalSingleOrList[RetAgent] = None,
    tolerance: OptionalSingleOrList[float] = None,
    area: OptionalSingleOrList[Area] = None,
    boundary: OptionalSingleOrList[Boundary] = None,
    time: OptionalSingleOrList[datetime] = None,
    weapon_name: OptionalSingleOrList[str] = None,
) -> list[Trigger]:
    """Create multiple triggers.

    Use the same parameters, or different parameters passed in as a list.

    If a list is passed in for any parameter its length must be equal to the number of
    triggers being created. If a single object is passed in for any parameter, that single
    object will be used for all triggers which take in the parameter.

    Args:
        number (int): The number of triggers to create.
        trigger_type (SingleOrListOptional[TriggerType]): The type of trigger for each trigger.
        sticky (SingleOrListOptional[bool]): The sticky parameter for each trigger. Defaults to
            False.
        log (SingleOrListOptional[bool]): The log parameter for each trigger. Defaults to True.
        invert (SingleOrListOptional[bool]): The invert parameter for each trigger. Defaults to
            False.
        position (OptionalSingleOrList[Coordinate2dOr3d]): The position parameter for each trigger.
            Required by KILLED_AGENTS_AT_POSITION and AGENT_AT_POSITION triggers. Defaults to None.
        agent (OptionalSingleOrList[RetAgent]): The agent parameter for each trigger. Required by
            AGENT_AT_POSITION, AGENT_IN_AREA, AGENT_NOT_IN_AREA, AGENT_CROSSED_BOUNDARY, and
            AGENT_MOVED_OUT_OF_AREA triggers. Defaults to None.
        tolerance (OptionalSingleOrList[float]): The tolerance parameter for each trigger. Required
            by AGENT_AT_POSITION trigger. Defaults to None.
        area (OptionalSingleOrList[Area]): The area parameter for each trigger. Required by
            AGENT_IN_AREA, AGENT_NOT_IN_AREA, and AGENT_MOVED_OUT_OF_AREA triggers. Defaults to
            None.
        boundary (OptionalSingleOrList[Boundary]): The boundary paramater for each trigger.
            Required by AGENT_CROSSED_BOUNDARY trigger. Defaults to None.
        time (OptionalSingleOrList[datetime]): The time paramater for each trigger. Required by
            TIME trigger. Defaults to None.
        weapon_name (OptionalSingleOrList[str]): The weapon name paramater for each trigger.
            Optionally required by AGENT_FIRED_WEAPON, WEAPON_FIRED_NEAR_LOCATION, and
            WEAPON_FIRED_NEAR_AGENT triggers.

    Returns:
        list[Trigger]: The created list of triggers.
    """
    _trigger_types: list[TriggerType] = create_list(number, trigger_type)

    trigger_config_lists = _create_configuration_lists(
        number, sticky, log, invert, position, agent, tolerance, area, boundary, time, weapon_name
    )

    trigger_list: list[Trigger] = []

    trigger_type_creators: dict[TriggerType, Callable] = {
        TriggerType.IMMEDIATE: _create_immediate_trigger,
        TriggerType.IMMEDIATE_SENSOR_FUSION: _create_immediate_sensor_fusion_trigger,
        TriggerType.KILLED_AGENTS_AT_POSITION: _create_killed_agents_at_position_trigger,
        TriggerType.ALIVE_AGENTS_AT_POSITION: _create_alive_agents_at_position_trigger,
        TriggerType.AGENT_AT_POSITION: _create_agent_at_position_trigger,
        TriggerType.AGENT_IN_AREA: _create_agent_in_area_trigger,
        TriggerType.AGENT_NOT_IN_AREA: _create_agent_not_in_area_trigger,
        TriggerType.AGENT_CROSSED_BOUNDARY: _create_agent_crossed_boundary_trigger,
        TriggerType.AGENT_MOVED_OUT_OF_AREA: _create_agent_moved_out_of_area_trigger,
        TriggerType.TIME: _create_time_trigger,
        TriggerType.AGENT_FIRED_WEAPON: _create_agent_fired_weapon_trigger,
        TriggerType.WEAPON_FIRED_NEAR_AGENT: _create_weapon_fired_near_agent_trigger,
        TriggerType.WEAPON_FIRED_NEAR_LOCATION: _create_weapon_fired_near_location_trigger,
        TriggerType.AGENT_KILLED: _create_agent_killed_trigger,
    }

    for trigger_counter in range(0, number):
        trigger_type_creators.get(
            _trigger_types[trigger_counter],
            _invalid_trigger_type,
        )(trigger_counter, trigger_config_lists, trigger_list)

    return trigger_list


def _check_arg_not_none(args: list[Any]) -> bool:
    """Checks that the required arguments for a trigger are not None.

    Args:
        args (list[Any]): A list of required arguments for a trigger.

    Raises:
        TypeError: Raises an error if any of the required arguments in the given list are None.

    Returns:
        bool: Returns True if all arguments are not None.
    """
    for arg in args:
        if arg is None:
            raise TypeError("Argument required for selected trigger not given.")

    return True


def _check_optional_arg_not_none(
    sticky: Optional[bool], log: Optional[bool], invert: Optional[bool]
) -> list[bool]:
    """Checks that the optional arguments for a trigger are not None.

    If an optional argument is none, the default for that argument is assigned. The default value
    for the sticky argument is False. The default value for the log argument is True. The default
    value for the Invert argument is False.

    Args:
        sticky (Optional[bool]): The 'sticky' optional argument for RET Triggers. Defaults to False
            if nothing is passed in.
        log (Optional[bool]): The 'log' optional argument for RET Triggers. Defaults to True if
            nothing is passed in.
        invert (Optional[bool]): The 'invert' optional argument for RET Triggers. Defaults to False
            if nothing is passed in.

    Returns:
        list[bool]: A list of the optional arguments for a trigger, with any None values replaced by
        the argument default value.
    """
    if sticky is None:
        sticky = False
    if log is None:
        log = True
    if invert is None:
        invert = False

    return [sticky, log, invert]
