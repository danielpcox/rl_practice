# Reinforcement Learning Practice

RL practice on the path to becoming an AGI researcher (as well as some public proof of life).
I'll be implementing algorithms and papers here.

Find me on Twitter [@danielpcox](https://twitter.com/danielpcox) if you see anything wrong or otherwise want to chat.

## Setup

```bash
# please read this script first and do something sensible
# virtualenv setup is commented out
./scripts/setup
```

## Usage

Train an algorithm with, e.g.:

```bash
python main.py train vpg
```

If you interrupt it with a KeyboardInterrupt exception (Ctrl+C), it'll save the model to `/tmp/agent.pt`.

Once you've got a trained agent saved somewhere, you can watch it play Pong with this:
```bash
python main.py run /tmp/agent.pt
```

