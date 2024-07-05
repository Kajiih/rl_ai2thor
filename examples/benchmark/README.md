# Experiments and Benchmark Tasks

## Experiments commands

### 1-item tasks

```bash
python examples/benchmark/train.py BreakBowl --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py BreakBowl --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py PickupPotato --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py PickupPotato --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py OpenBook --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py OpenBook --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py SwitchOnTV --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py SwitchOnTV --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py OpenToilet --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py OpenToilet --model PPO --log-metrics --eval --seed 0 --no-adv
```

### 2-items tasks

```bash
python examples/benchmark/train.py PourCoffee --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py PourCoffee --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py LookBookInLight --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py LookBookInLight --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py PlacePotatoInFridge --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py PlacePotatoInFridge --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py PlaceNewspaperOnSofa --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py PlaceNewspaperOnSofa --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py BringTowelClothesClose --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py BringTowelClothesClose --model PPO --log-metrics --eval --seed 0 --no-adv
```

### >3-items tasks

```bash

python examples/benchmark/train.py PlaceTomatoPotatoInFridge --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py PlaceTomatoPotatoInFridge --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py WatchTV --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py WatchTV --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py PlacePenBookOnDesk --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py PlacePenBookOnDesk --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py ReadBookInBedSimple --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py ReadBookInBedSimple --model PPO --log-metrics --eval --seed 0 --no-adv

python examples/benchmark/train.py SetupBathSimple --model PPO --log-metrics --eval --seed 0

python examples/benchmark/train.py SetupBathSimple --model PPO --log-metrics --eval --seed 0 --no-adv
```

## Benchmark tasks

### PrepareMeal

```bash
python examples/benchmark/train.py PrepareMeal --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py PrepareMeal --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py PrepareMeal --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### CleanUpKitchen

```bash
python examples/benchmark/train.py CleanUpKitchen --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py CleanUpKitchen --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py CleanUpKitchen --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### RelaxOnSofa

```bash
python examples/benchmark/train.py RelaxOnSofa --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py RelaxOnSofa --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py RelaxOnSofa --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### CleanUpLivingRoom

```bash
python examples/benchmark/train.py CleanUpLivingRoom --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py CleanUpLivingRoom --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py CleanUpLivingRoom --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### ReadBookInBed

```bash
python examples/benchmark/train.py ReadBookInBed --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py ReadBookInBed --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py ReadBookInBed --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### CleanUpBedroom

```bash
python examples/benchmark/train.py CleanUpBedroom --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0 --no-adv
python examples/benchmark/train.py CleanUpBedroom --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0 --no-adv
python examples/benchmark/train.py CleanUpBedroom --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0 --no-adv
```

### SetupBath

```bash
python examples/benchmark/train.py SetupBath --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py SetupBath --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py SetupBath --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### CleanUpBathroom

```bash
python examples/benchmark/train.py CleanUpBathroom --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0 --no-adv
python examples/benchmark/train.py CleanUpBathroom --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0 --no-adv
python examples/benchmark/train.py CleanUpBathroom --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0 --no-adv
```

### MultiTask4

```bash
python examples/benchmark/train.py MultiTask4 --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py MultiTask4 --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py MultiTask4 --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```

### MultiTask8

```bash
python examples/benchmark/train.py MultiTask8 --model Random -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py MultiTask8 --model PPO -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
python examples/benchmark/train.py MultiTask8 --model QRDQN -s 100000 --randomize-agent --nb-scenes 1 --group --seed 0
```
