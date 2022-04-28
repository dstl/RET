"""Agent orders, tasks and triggers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from mesa_ret.template import Template

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent


class TaskLogStatus(Enum):
    """Task logging statuses."""

    STARTED = "Started"
    FINISHED = "Finished"
    INTERRUPTED = "Interrupted"


class TriggerLogStatus(Enum):
    """Trigger logging statuses."""

    ACTIVATED = "Activated"
    DEACTIVATED = "Deactivated"


class Task(ABC, Template):
    """An abstract class representing a task that an agent can perform.

    This class is abstract and should be overridden by classes with specific behaviour
    as and when required. They contain both the instructions on what the agent should
    do to undertake the task and also the logic to determine if the task has been
    completed.

    To create a new task using custom logic create a new task class that extends this
    class and then override the following methods:

        _do_task_step: This must be overridden and should call the relevant method on
            the agent undertaking the task, calculating and passing in the required
            parameters as required.
        _is_task_complete: This must be overridden and should contain the logic to
            determine if the task has been completed.
        get_new_instance: This must be overridden and should return a new instance of
            your task that is functionally identical. This is normally achieved by
            creating a new instance of your class with the same input parameters.
            This is required to ensure that every order has it's own instance of the
            task to avoid the actions of one agent affecting the triggers of other
            orders (as tasks are allowed to contain state and be mutable).
        __str__: Overriding this method is optional, but recommended, and should
            provide a human readable name for the task that is used for logging.
            By convention it should end with the word "Task" e.g. "Move Task".
        _get_log_message: Overriding this method is optional, it is used to provide
            extra information about the the task to the logging framework,
            e.g. the destination for the move task. If not overridden no extra
            information will be logged.

    For examples of tasks and how to override these methods see the tasks already
    implemented.
    """

    _started: bool
    _complete: bool
    _log: bool

    def __init__(self, log: bool) -> None:
        """Create a task, override as necessary in subclasses.

        Args:
            log (bool): whether to log or not.
        """
        self._started = False
        self._complete = False
        self._log = log

    def do_task_step(self, doer: RetAgent) -> None:
        """Do one time steps worth of the task and log.

        Args:
            doer (RetAgent): the agent that will do the task
        """
        if not self._started:
            self._log_task(doer, TaskLogStatus.STARTED)
            self._started = True
        self._do_task_step(doer)

    def is_task_complete(self, doer: RetAgent) -> bool:
        """Check whether the task is complete and log.

        Args:
            doer (RetAgent): the agent that will do the task

        Returns:
            bool: true when the task is completed, false otherwise
        """
        if not self._complete:
            self._complete = self._is_task_complete(doer)
            if self._complete:
                self._log_task(doer, TaskLogStatus.FINISHED)
        return self._complete

    @abstractmethod
    def _do_task_step(self, doer: RetAgent) -> None:  # pragma: no cover
        """Do one time steps worth of the task, override in subclasses.

        Args:
            doer (RetAgent): the agent that will do the task
        """
        pass

    @abstractmethod
    def _is_task_complete(self, doer: RetAgent) -> bool:  # pragma: no cover
        """Check whether the task is complete, override in subclass.

        Args:
            doer (RetAgent): The agent completing the task

        Returns:
            bool: true when the task is completed, false otherwise
        """
        pass

    @abstractmethod
    def get_new_instance(self) -> Task:  # pragma: no cover
        """Return a new instance of a functionally identical task.

        This abstract method is only defined here, rather than just inherited from
        the Template class, to provide more specific type hints.

        Returns:
            Task: New instance of the task
        """
        pass

    def _log_task(self, doer: RetAgent, status: TaskLogStatus) -> None:
        """Log tasks.

        Args:
            doer (RetAgent): The agent performing the task
            status (TaskLogStatus): Status of the task
        """
        if self._log:
            doer.model.logger.log_task(doer, str(self), status, self._get_log_message())

    def _get_log_message(self) -> str:
        """Get the log message, override in subclasses.

        Returns:
            str: The log message
        """
        return ""


class CompoundTask(Task):
    """A class representing a list of tasks that an agent can perform."""

    _task_templates: list[Template[Task]]
    _remaining_tasks: list[Task]

    def __init__(self, tasks: list[Template[Task]], log: bool = True) -> None:
        """Create a compound task.

        Args:
            tasks (list[Task]): the list of tasks in order that this compound task will
                initiate
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)
        self._task_templates = tasks
        self._remaining_tasks = [t.get_new_instance() for t in tasks]

    def __str__(self):
        """Output a human readable list of tasks in the compound task.

        Returns:
            string: brief description of the compound task
        """
        task_list = map(lambda a: str(a), self._task_templates)
        return "Compound Task: (" + ", ".join(task_list) + ")"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time steps worth of the next task.

        Args:
            doer (RetAgent): the agent that will do the task
        """
        if len(self._remaining_tasks) > 0:
            current_task = self._remaining_tasks[0]
            current_task.do_task_step(doer)

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check whether compound task is complete.

        Args:
            doer (RetAgent): the agent that will do the task

        Returns:
            bool: true when all tasks are completed, false otherwise
        """
        if len(self._remaining_tasks) > 0:
            current_task = self._remaining_tasks[0]
            if current_task.is_task_complete(doer):
                self._remaining_tasks.remove(current_task)

        return len(self._remaining_tasks) == 0

    def get_new_instance(self) -> CompoundTask:
        """Return a new instance of a functionally identical compound task.

        Returns:
            CompoundTask: New instance of the compound task
        """
        return CompoundTask(tasks=self._task_templates, log=self._log)


class Trigger(ABC, Template):
    """A condition that can be used to identify when orders are active.

    This class is abstract and should be overridden by classes with specific behaviour
    as and when required.

    To create a new trigger using custom logic create a new trigger class that extends
    this class and then override the following methods:

        _check_condition: This must be overridden and should contain the logic to
            determine when your trigger should be active.
        get_new_instance: This must be overridden and should return a new instance of
            your trigger that is functionally identical. This is normally achieved by
            creating a new instance of your class with the same input parameters.
            This is required to ensure that every order has it's own instance of the
            trigger to avoid the actions of one agent affecting the triggers of other
            orders (as triggers are allowed to contain state and be mutable).
        __str__: Overriding this method is optional, but recommended, and should
            provide a human readable name for the trigger that is used for logging.
            By convention it should end with the word "Trigger" e.g. "Position Trigger".
        _get_log_message: Overriding this method is optional, it is used to provide
            extra information about the the trigger to the logging framework,
            e.g. the time that the trigger is configured to activate for the
            time trigger. If not overridden no extra information will be logged.

    For examples of triggers and how to override these methods see the triggers
    already implemented.
    """

    _log: bool
    _sticky: bool
    _active_flag: bool = False
    _invert: bool

    def __init__(self, log: bool, sticky: bool = False, invert: bool = False) -> None:
        """Create a trigger.

        Args:
            log (bool): whether to log or not.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        self._log = log
        self._sticky = sticky
        self._invert = invert

    def is_active(self, checker: RetAgent) -> bool:
        """Check if this trigger is active.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if active
        """
        if self._sticky and self._active_flag:
            pass
        else:
            activated = self._check_condition(checker) != self._invert

            if activated != self._active_flag:
                if activated:
                    self._log_trigger(checker, TriggerLogStatus.ACTIVATED)
                else:
                    self._log_trigger(checker, TriggerLogStatus.DEACTIVATED)

            self._active_flag = activated

        return self._active_flag

    @abstractmethod
    def _check_condition(self, checker: RetAgent) -> bool:  # pragma: no cover
        """Abstract method that should be overridden with the logic for the trigger.

        Args:
            checker (RetAgent): the agent doing the checking, the agents perceived
                world should be used for world information

        Returns:
            bool: true if condition is true
        """
        pass

    def _log_trigger(self, checker: RetAgent, status: TriggerLogStatus) -> None:
        """Log trigger.

        Args:
            checker (RetAgent): The agent performing the check
            status (TriggerLogStatus): The status of the trigger
        """
        if self._log:
            checker.model.logger.log_trigger(checker, str(self), status, self._get_log_message())

    def _get_log_message(self) -> str:
        """Get the log message, override in subclasses.

        Returns:
            str: The log message
        """
        return ""

    @abstractmethod
    def get_new_instance(self) -> Trigger:  # pragma: no cover
        """Return a new instance of a functionally identical trigger.

        This abstract method is only defined here, rather than just inherited from
        the Template class, to provide more specific type hints.

        Returns:
            Trigger: New instance of the trigger
        """
        pass

    def get_new_instance_sticky_status_maintained(self) -> Trigger:
        """Return a new instance of a functionally identical trigger.

        If sticky the active_flag state is retained

        Returns:
            Trigger: New instance of the trigger
        """
        new_instance = self.get_new_instance()
        if self._sticky and self._active_flag:
            new_instance._active_flag = True
        return new_instance


class CompoundTrigger(Trigger, ABC):
    """An abstract class for a set of triggers.

    A compound trigger must have a condition which determines whether
    the compound of all triggers is active.
    """

    _triggers: list[Template[Trigger]]

    def __init__(
        self,
        triggers: list[Template[Trigger]],
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create a compound trigger.

        Args:
            triggers (list[Template[Trigger]]): List of triggers to be checked
            sticky (bool): if true once activated this compound trigger will remain
                activated. Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._triggers = [t.get_new_instance() for t in triggers]

    def __str__(self) -> str:
        """Output a human readable string to describe the compound trigger.

        Returns:
            str: brief description of the compound trigger
        """
        return "Compound Trigger: (" + self._get_triggers_as_str() + ")"

    def _get_triggers_as_str(self) -> str:
        """Output a human readable list of triggers in the compound trigger.

        Returns:
            str: a list of triggers in the compound trigger
        """
        trigger_list = map(lambda a: str(a), self._triggers)
        return ", ".join(trigger_list)

    @abstractmethod
    def _check_condition(self, checker: RetAgent) -> bool:  # pragma: no cover
        """Check status of all triggers.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the correct combination of triggers are active
        """
        pass


class CompoundAndTrigger(CompoundTrigger):
    """A set of triggers that must all be active for the compound to be active."""

    def __init__(
        self,
        triggers: list[Template[Trigger]],
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create a compound 'and' trigger.

        Args:
            triggers (list[Trigger]): List of triggers to be checked
            sticky (bool): if true once activated this compound trigger will remain
                activated. Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(triggers=triggers, sticky=sticky, log=log, invert=invert)

    def __str__(self):
        """Output a human readable list of triggers in the compound 'and' trigger.

        Returns:
            string: brief description of the compound trigger
        """
        return "Compound 'And' Trigger: (" + self._get_triggers_as_str() + ")"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check status of all triggers.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if all triggers are active
        """
        return all(trigger.is_active(checker) for trigger in self._triggers)  # type: ignore

    def get_new_instance(self) -> CompoundAndTrigger:
        """Return a new instance of a functionally identical compound 'and' trigger.

        Returns:
            CompoundAndTrigger: New instance of the compound 'and' trigger
        """
        return CompoundAndTrigger(
            triggers=self._triggers, sticky=self._sticky, invert=self._invert, log=self._log
        )


class CompoundOrTrigger(CompoundTrigger):
    """Set of triggers, at least one must be active for the compound to be active."""

    def __init__(
        self,
        triggers: list[Template[Trigger]],
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create a compound 'or' trigger.

        Args:
            triggers (list[Trigger]): List of triggers to be checked
            sticky (bool): if true once activated this compound trigger will remain
                activated. Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(triggers=triggers, sticky=sticky, log=log, invert=invert)

    def __str__(self):
        """Output a human readable list of triggers in the compound 'or' trigger.

        Returns:
            string: brief description of the compound trigger
        """
        return "Compound 'Or' Trigger: (" + self._get_triggers_as_str() + ")"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check status of all triggers.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if at least one trigger is active
        """
        return any(trigger.is_active(checker) for trigger in self._triggers)  # type: ignore

    def get_new_instance(self) -> CompoundOrTrigger:
        """Return a new instance of a functionally identical compound 'or' trigger.

        Returns:
            CompoundOrTrigger: New instance of the compound 'or' trigger
        """
        return CompoundOrTrigger(
            triggers=self._triggers, sticky=self._sticky, invert=self._invert, log=self._log
        )


class Order(Template):
    """Combination of trigger and task, along with a persistent flag and a priority."""

    _trigger: Trigger
    _task: Task
    _persistent: bool

    priority: int

    def __init__(
        self,
        trigger: Template[Trigger],
        task: Template[Task],
        persistent: bool = False,
        priority: int = 0,
    ) -> None:
        """Create an order.

        Args:
            trigger (Template[Trigger]): template for the trigger for the condition(s)
                that must met to activate this order
            task (Template[Task]): template for the task to be completed on activation
                of the order
            persistent (bool): if true the order can be activated multiple time, if false
                once the order has been completed it will not be undertaken again.
                Defaults to False.
            priority (int): higher priority tasks take preference over lower priority
                tasks. Defaults to 0.
        """
        self._trigger = trigger.get_new_instance()
        self._task = task.get_new_instance()
        self._persistent = persistent

        self.priority = priority

    def __str__(self):
        """Output a human readable list of tasks and triggers in the order.

        Returns:
            string: brief description of the order
        """
        return str(self._task) + " and " + str(self._trigger)

    def get_order_string(self):
        """Generate human readable string for the order.

        Returns:
            str: the order task and the order trigger.
        """
        return str(self._task) + ": " + str(self._trigger)

    def is_persistent(self) -> bool:
        """Check if order is persistent.

        Returns:
            bool: true if persistent
        """
        return self._persistent

    def is_condition_met(self, checker: RetAgent) -> bool:
        """Check if order condition has been met.

        Args:
            checker (RetAgent): agent doing the checking

        Returns:
            bool: true if condition is met
        """
        return self._trigger.is_active(checker)

    def execute_step(self, doer: RetAgent) -> None:
        """Execute a time step of the task.

        Args:
            doer (RetAgent): agent doing the task
        """
        self._task.do_task_step(doer)

    def is_complete(self, doer: RetAgent) -> bool:
        """Check if a task is completed.

        Args:
            doer (RetAgent): agent doing the task

        Returns:
            bool: true if the task has been completed
        """
        if self._task.is_task_complete(doer):
            return True
        else:
            return False

    def get_new_instance(self) -> Order:
        """Return a new instance of a functionally identical order.

        Returns:
            Order: New instance of the order
        """
        return Order(
            trigger=self._trigger,
            task=self._task,
            persistent=self._persistent,
            priority=self.priority,
        )

    def get_new_instance_sticky_status_maintained(self) -> Order:
        """Return a new instance of a functionally identical order.

        If the trigger is sticky it's status is kept.

        Returns:
            Order: New instance of the trigger
        """
        new_instance = self.get_new_instance()
        new_instance._trigger = self._trigger.get_new_instance_sticky_status_maintained()
        return new_instance


class GroupAgentOrder(Order):
    """A specific type of order which is sent directly to a group agent.

    Only tasks which are to be completed by a GroupAgent should be part of a GroupAgentOrder.
    Other orders which are passed to a GroupAgent will be disseminated to it's
    subordinate agents. This order type allows an order to be passed to a group agent
    directly instead of being automatically passed on to it's subordinate agents.
    """

    def get_new_instance(self) -> GroupAgentOrder:
        """Return a new instance of a functionally identical group order.

        Returns:
            Order: New instance of the group order
        """
        return GroupAgentOrder(
            trigger=self._trigger,
            task=self._task,
            persistent=self._persistent,
            priority=self.priority,
        )

    def get_new_instance_sticky_status_maintained(self) -> GroupAgentOrder:
        """Return a new instance of a functionally identical group order.

        If the trigger is sticky it's status is kept.

        Returns:
            Order: New instance of the trigger
        """
        new_instance = self.get_new_instance()
        new_instance._trigger = self._trigger.get_new_instance_sticky_status_maintained()
        return new_instance
