# rl_library
A library containing all of the RL algorithms I've implemented and example applications. The repository is broken up as 
follows and explained below:
1. algorithms
2. runners
3. utils

# algorithms


## Augmented Random Search (ARS)

# runners
An interface between the environment and the learning agent. Each runner follows inherits the following abstract methods 
and form:

```python
from runners.abstract_runner import Runner
class GenericRunner(Runner):
    """
    A runner works as an interface between the learning agent and the learning environment. Anything the agent wants to
    do in the environment should be run through a runner. Each environment should gets its own style of runner because
    every environment operates differently.
    
    Any method not included in this list should be preceded with __ to denote that is is unique to this specific
    runner. e.g.
        def __other_method(self): return
    """

    @abstractmethod
    def get_state(self):
        """
        This method should return the current state of the agent in the environment. No reward or status of done or
        exit will be provided, just the state.

        :input:
            None
        :output:
            return curr_state
        """
        pass

    @abstractmethod
    def step(self):
        """
        This method should execute a single step within the environment and return all necessary information
        including in the following order:
            1 next state/observation
            2 reward
            3 done (if the agent has reached a terminal state, this will be 1, otherwise 0)
            4 exit condition (if the agent has reached a fatal state, this will be 1, otherwise 0)

        :input:
            action
        :output:
            return next_state, reward, done, exit
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method should reset the environment. In the case of a simulation, this should take the agent back to a
        safe starting point. In a real-world system, this might involve halting until a resume signal has been sent
        allowing the user to move the agent to a safe starting location.

        :output:
            Nothing is returned from this function.
        """
        pass
```


# utils
