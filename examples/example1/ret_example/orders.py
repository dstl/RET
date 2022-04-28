"""Orders for the Example model."""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from mesa_ret.creator.addorders import add_background_orders_to_agents, add_orders_to_agents
from mesa_ret.creator.triggers import TriggerType, create_triggers
from mesa_ret.orders.background_order import BackgroundOrder
from mesa_ret.orders.order import CompoundOrTrigger, CompoundTask, Order, Trigger
from mesa_ret.orders.tasks.act_and_move import ActAndMoveTask
from mesa_ret.orders.tasks.fire import (
    DetermineTargetAndFireTask,
    FireAtAreaTask,
    FireAtTargetTask,
    NamedWeaponSelector,
)
from mesa_ret.orders.tasks.move import FixedRandomMoveTask, MoveInBandTask, MoveTask
from mesa_ret.orders.tasks.sense import SenseTask
from mesa_ret.orders.triggers.immediate import ImmediateTrigger
from mesa_ret.orders.triggers.killed import AgentKilledTrigger
from mesa_ret.orders.triggers.weapon import AgentFiredWeaponTrigger
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.perceivedworld import (
    And,
    FriendlyAgents,
    HostileAgents,
    IdentifiedAgents,
    RecognisedAgents,
)

from . import constants

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.model import RetModel


class FriendlyRecognisedAgentsTrigger(Trigger):
    """Friendly recognised agents trigger.

    Resolves to true if there are any alive friendly agents at recognised level in the checker's
    perceived world.
    """

    def __init__(self, log=True, sticky=False):
        """Initialise trigger.

        Args:
            log (bool): whether to log or not.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
        """
        super().__init__(log, sticky)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Friendly Recognised Agents Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for identified friendly agents.

        Args:
            checker (RetAgent): the agent doing the checking, the agents perceived
                world should be used for world information

        Returns:
            bool: true if condition is true
        """
        friendly_agents = checker.perceived_world.get_perceived_agents(
            And([FriendlyAgents(), RecognisedAgents()])
        )
        alive_friendly_agents = [
            agent for agent in friendly_agents if agent.casualty_state != AgentCasualtyState.KILLED
        ]
        return len(alive_friendly_agents) > 0

    def get_new_instance(self) -> FriendlyRecognisedAgentsTrigger:
        """Get new instance of Trigger.

        Returns:
            FriendlyRecognisedAgentsTrigger: New instance of the object
        """
        return FriendlyRecognisedAgentsTrigger(log=self._log, sticky=self._sticky)


class HostileIdentifiedAgentsTrigger(Trigger):
    """Hostile identified agents trigger."""

    def __init__(self, log=True, sticky=False):
        """Initialise trigger.

        Args:
            log (bool): whether to log or not.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
        """
        super().__init__(log, sticky)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Hostile Identified Agents Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for identified hostile agents.

        Args:
            checker (RetAgent): the agent doing the checking, the agents perceived
                world should be used for world information

        Returns:
            bool: true if condition is true
        """
        hostile_agents = checker.perceived_world.get_perceived_agents(
            And([HostileAgents(), IdentifiedAgents()])
        )
        return len(hostile_agents) > 0

    def get_new_instance(self) -> HostileIdentifiedAgentsTrigger:
        """Get new instance of Trigger.

        Returns:
            HostileIdentifiedAgentsTrigger: New instance of the object
        """
        return HostileIdentifiedAgentsTrigger(log=self._log, sticky=self._sticky)


class ExampleOrdersController:
    """Example orders controller."""

    @staticmethod
    def set_friendly_orders(model: RetModel, agents: list[RetAgent]) -> None:
        """Set friendly agent orders.

        See the tutorial overview for full details but in summary:

        - Rockets fire into area where AD agents is
        - UAV flies over to see if its dead
          - If the UAV dies OR it sees a living AD unit then fire again at that location with the
            rockets
        - Armoured Inf advance towards urban area once rockets have fired
        - When Armoured inf enter the defended area the rockets fire into the urban area

        Args:
            model (RetModel): The RET model.
            agents (list[RetAgent]): list of friendly agents to set orders for.
        """
        # ai_coy_hq = [a for a in agents if a.name == "Blue AI Coy"][0]
        ai_pl_1_sects = [a for a in agents if "Blue AI Sect 1" in a.name]
        ai_pl_2_sects = [a for a in agents if "Blue AI Sect 2" in a.name]
        ai_pl_3_sects = [a for a in agents if "Blue AI Sect 3" in a.name]
        # ai_pl_1 = [a for a in agents if a.name == "Blue AI Pl 1"][0]
        # ai_pl_2 = [a for a in agents if a.name == "Blue AI Pl 2"][0]
        # ai_pl_3 = [a for a in agents if a.name == "Blue AI Pl 3"][0]
        rocket_arty = [a for a in agents if a.name == "Blue Rocket Arty"][0]
        uavs = [a for a in agents if "Blue UAV" in a.name]
        background_orders = [
            BackgroundOrder(
                time_period=timedelta(seconds=30), task=SenseTask(duration=timedelta(hours=15))
            )
        ]
        air_def = [a for a in agents if a.name == "Red Air Defence"][0]
        rocket_orders = [
            Order(
                trigger=AgentKilledTrigger(air_def, invert=True),  # Fire while agent not killed
                task=FireAtAreaTask(
                    target_area=constants.ad_agent_area,
                    rounds=24,
                    weapon_selector=NamedWeaponSelector(name="Rocket"),
                    random=model.random,
                ),
                priority=1,
                persistent=True,  # keep firing until the agent is killed
            ),
            # TODO: To be entirely credible, this wouldn't be a perception that these agents had
            #  entered the area but a message from them saying they had.
            Order(
                trigger=CompoundOrTrigger(
                    create_triggers(
                        number=len(ai_pl_1_sects),
                        trigger_type=TriggerType.AGENT_IN_AREA,
                        agent=ai_pl_1_sects,
                        sticky=False,
                        area=constants.urban_approach,
                    )
                ),
                task=FireAtTargetTask(
                    target=constants.urban_centre,
                    rounds=24,
                    weapon_selector=NamedWeaponSelector(name="Rocket"),
                ),
                priority=1,
                persistent=False,
            ),
        ]

        add_orders_to_agents(rocket_arty, rocket_orders)

        uav_orders = [
            Order(
                trigger=ImmediateTrigger(),
                task=CompoundTask(
                    [
                        MoveInBandTask(
                            destination=constants.ad_agent_area_uav_drop_down_pos,
                            bands=constants.uav_height_bands,
                            space=model.space,
                            tolerance=constants.uav_position_tolerance,
                        ),
                        MoveInBandTask(
                            destination=constants.ad_agent_area_centre_3d,
                            bands=constants.uav_height_bands,
                            space=model.space,
                            tolerance=constants.uav_position_tolerance,
                        ),
                    ]
                ),
                priority=1,
                persistent=False,
            )
        ]

        add_orders_to_agents(uavs, uav_orders)
        add_background_orders_to_agents(uavs, background_orders)

        combat_sect_orders = [
            Order(
                trigger=AgentFiredWeaponTrigger(firer=rocket_arty, weapon_name="Rocket"),
                task=CompoundTask(
                    [
                        MoveTask(
                            destination=constants.approach_dogleg_position,
                            tolerance=constants.ground_unit_position_tolerance,
                        ),
                        MoveTask(
                            destination=constants.urban_centre,
                            tolerance=constants.ground_unit_position_tolerance,
                        ),
                    ]
                ),
                priority=1,
                persistent=False,
            ),
            Order(
                trigger=HostileIdentifiedAgentsTrigger(),
                task=DetermineTargetAndFireTask(rounds=4),
                priority=2,
                persistent=True,
            ),
        ]

        add_orders_to_agents(ai_pl_1_sects, combat_sect_orders)
        add_orders_to_agents(ai_pl_2_sects, combat_sect_orders)
        add_orders_to_agents(ai_pl_3_sects, combat_sect_orders)
        add_background_orders_to_agents(ai_pl_1_sects, background_orders)
        add_background_orders_to_agents(ai_pl_2_sects, background_orders)
        add_background_orders_to_agents(ai_pl_3_sects, background_orders)

    @staticmethod
    def set_hostile_orders(model: RetModel, agents: list[RetAgent]):
        """Set orders for hostile agents.

        Args:
            model (RetModel): The RET model.
            agents (list[RetAgent]): A list of hostile agents to set orders for.
        """
        mech_inf = [a for a in agents if "Red Mech Inf Sect" in a.name]
        anti_armour = [a for a in agents if a.name == "Red Anti Armour"][0]
        air_def = [a for a in agents if a.name == "Red Air Defence"][0]
        ai_pl_1_sects = [a for a in agents if "Blue AI Sect 1" in a.name]
        ai_pl_2_sects = [a for a in agents if "Blue AI Sect 2" in a.name]
        ai_pl_3_sects = [a for a in agents if "Blue AI Sect 3" in a.name]

        background_orders = [
            BackgroundOrder(
                time_period=timedelta(seconds=30), task=SenseTask(duration=timedelta(hours=15))
            )
        ]

        anti_armour_orders = [
            Order(
                trigger=CompoundOrTrigger(
                    create_triggers(
                        number=len(ai_pl_1_sects + ai_pl_2_sects + ai_pl_3_sects),
                        trigger_type=TriggerType.AGENT_CROSSED_BOUNDARY,
                        agent=ai_pl_1_sects + ai_pl_2_sects + ai_pl_3_sects,
                        boundary=constants.ambush_line,
                    ),
                    sticky=True,
                ),
                task=DetermineTargetAndFireTask(),
                priority=1,
                persistent=True,
            )
        ]
        add_orders_to_agents(anti_armour, anti_armour_orders)
        add_background_orders_to_agents(anti_armour, background_orders)

        air_def_move_task = FixedRandomMoveTask(
            min_displacement=50,
            max_displacement=1500,
            x_min=model.space.x_min,
            x_max=model.space.x_max,
            y_min=model.space.y_min,
            y_max=model.space.y_max,
            tolerance=5,
            area=constants.ad_agent_area,
        )
        air_def_fire_task = DetermineTargetAndFireTask(rounds=30)
        air_def_fire_and_move_task = ActAndMoveTask(
            action_task=air_def_fire_task, move_task=air_def_move_task
        )

        air_def_orders = [
            Order(
                trigger=FriendlyRecognisedAgentsTrigger(),
                task=air_def_fire_and_move_task,
                priority=1,
                persistent=True,
            )
        ]
        add_orders_to_agents(air_def, air_def_orders)
        add_background_orders_to_agents(air_def, background_orders)

        mech_inf_move_task = FixedRandomMoveTask(
            min_displacement=50,
            max_displacement=150,
            x_min=model.space.x_min,
            x_max=model.space.x_max,
            y_min=model.space.y_min,
            y_max=model.space.y_max,
            tolerance=5,
        )
        mech_inf_fire_task = DetermineTargetAndFireTask(rounds=5)
        mech_inf_fire_and_move_task = ActAndMoveTask(
            action_task=mech_inf_fire_task, move_task=mech_inf_move_task
        )

        mech_inf_orders = [
            Order(
                trigger=FriendlyRecognisedAgentsTrigger(),
                task=mech_inf_fire_and_move_task,
                priority=1,
                persistent=True,
            )
        ]

        add_orders_to_agents(agents=mech_inf, orders=mech_inf_orders)
        add_background_orders_to_agents(mech_inf, background_orders)
