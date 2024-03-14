#!/usr/bin/env bash
# run this script in the root path of DAR-MVSNet
MVS_TRAINING="/home/ubuntui/dtu_training/" # path to dataset mvs_training
LOG_DIR="./outputs/dtu_training" # path to checkpoints
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

NGPUS=1
BATCH_SIZE=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS /home/ubuntu/litingshuai/DAR-MVSNet-master/train.py \
	--logdir=$LOG_DIR \
	--dataset=dtu_yao \
	--batch_size=$BATCH_SIZE \
	--epochs=10 \
	--trainpath=$MVS_TRAINING \
	--trainlist=/home/ubuntu/litingshuai/DAR-MVSNet-master/lists/dtu/train.txt \
	--testlist=/home/ubuntu/litingshuai/DAR-MVSNet-master/lists/dtu/val.txt \
	--numdepth=192 \
	--ndepths="48,32,8" \
	--nviews=5 \
	--wd=0.0001 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="6,8:2" \
	--dlossw="1.0,1.0,1.0" | tee -a $LOG_DIR/log.txt
