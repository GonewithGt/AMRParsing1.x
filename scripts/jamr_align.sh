#!/bin/bash


#### Config ####
. ~/jamr/scripts/config.sh

#### Align the tokenized amr file ####

echo "### Aligning $1 ###"
# input should be tokenized AMR file, which has :tok tag in the comments
~/jamr/scripts/ALIGN.sh -v 0 < $1 > $1.aligned
