#!/bin/bash
set +e

MODEL_BASENAME=models/03281310_7l_solar \
MAX_WALLCLOCK_SECONDS=4000 \
SOLAR_WALLCLOCK_SECONDS=4000 \
NUM_LAYERS=7 \
TRAIN_BATCH_TOKENS=524288 \
SOLAR_WARMDOWN=1000 \
SOLAR_LAYER_PATTERN=1,2,3,4,1,4 \
SOLAR_DECODER_ONLY=1 \
SOLAR_RESUME_PATH=models/03281310_7l_stage1.pt \
uv run torchrun --standalone --nproc_per_node=2 train_solar.py

MODEL_BASENAME=models/03281310_8l_solar \
MAX_WALLCLOCK_SECONDS=4000 \
SOLAR_WALLCLOCK_SECONDS=4000 \
NUM_LAYERS=8 \
TRAIN_BATCH_TOKENS=524288 \
SOLAR_WARMDOWN=1000 \
SOLAR_LAYER_PATTERN=1,2,3,3,4 \
SOLAR_DECODER_ONLY=0 \
SOLAR_RESUME_PATH=models/03281310_8l_stage1.pt \
uv run torchrun --standalone --nproc_per_node=2 train_solar.py

we would like another option for solar, so that when SOLAR_DECODER_ONLY=1 we can either reuse skip connections, so that duplicated layer gets same skip as layer from which it was duplicated or that duplicated wont get any skip. We would like to use SOLAR_NEW_LAYERS_USE_SKIP as a new envvar


We will use * to indicate new layers (duplicated, 3rd occurance, etc.)
for pattern 1,2,3,4
the full pattern 1,2,3,4,5,6,7

for pattern 1,2,3,4,4
the full pattern 1,2,3,4,5,6,7,7*

for pattern 1,2,3,3,4 and SOLAR_DECODER_ONLY=0
the full pattern 1,2,3,3*,4,5,6,6*,7* (3* has skip to 6*)

for pattern 1,2,3,3,4 and SOLAR_DECODER_ONLY=1 SOLAR_NEW_LAYERS_USE_SKIP=1
the full pattern 1,2,3,4,5,6,6*,7 (3 has skip to 6 and 6*)

for pattern 1,2,3,3,4 and SOLAR_DECODER_ONLY=1 SOLAR_NEW_LAYERS_USE_SKIP=0
the full pattern 1,2,3,4,5,6,6*,7 (3 has skip ONLY to 6, 6* has no skip at all)

Are the patterns clear for you? If so give me other examples with not even number of layers, otherwise ask questions?


We have UNet like arch of gpt. so if we have 6 layers in total then there is 3 encoder layers followed by 3 decoder layers. There are skip connections between encoders and decoders. If number is not even then we get another decoder layer at the end, which does NOT have corresponding encoder layer which results in no skip connection for that layer.

We are running a lot of experiments with layers duplications, sometimes we want to duplicate only decoders, sometimes both. We are using patterns to specify which layers do duplicate. For example if we have 4 layers in total and we specify pattern 1,2,2 then we will get in total 6 layers, cause last layer of encoder and last layer of decoder will get duplicated, so the full pattern would look like 1,2,2,3,4,4. Its get a little trickier when we have uneven number of layers, for example 3 and we specify pattern 1,2,2 then onbviously encoder has no 2nd layer, so we will be duplicating only last decoder layer, so we will end up with full pattern 1,2,3,3.

Is that clear for you? Could you show me some patterns and then respective full patterns with explanations?

WANDB_ENABLED=0 MODEL_BASENAME=models/03281310_7l_solar \
VAL_LOSS_EVERY=100 \
TRAIN_LOG_EVERY=50 \
SOLAR_WARMUP=100 \
SOLAR_WARMDOWN=0 \
MAX_WALLCLOCK_SECONDS=1000 \
SOLAR_WALLCLOCK_SECONDS=1000 \
NUM_LAYERS=7 \
TRAIN_BATCH_TOKENS=524288 \
SOLAR_WARMDOWN=1000 \
SOLAR_DUP_TYPE="full" \
SOLAR_LAYER_PATTERN=1,2,3,4,1,4 \
SOLAR_DECODER_ONLY=1 \
SOLAR_RESUME_PATH=models/03281310_7l_stage1.pt \
uv run torchrun --standalone --nproc_per_node=2 train_solar.py