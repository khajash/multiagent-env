# multiagent-env

This multi-agent pybox2d environment is based off of OpenAI's [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs). 
The files that are custom for my environment are:
* ./multiagent/my_core.py
* ./multiagent/my_environment.py
* ./multiagent/my_world.py
* ./multiagent/scenarios/simple_blocks.py

***NOTE: this project is in progress***

## Setup
- Recommended to use a virtual environment, such as `venv`, `virtualenv` or `conda`

```
git clone https://github.com/khajash/multiagent-env.git
cd multiagent-env
python -m venv .env
source .env/bin/activate
pip install -e .
```

## Usage
- Run `python test_env.py` to see the agents run using a random policy. 
    - Look at `test_env.py` for more details on usage.
