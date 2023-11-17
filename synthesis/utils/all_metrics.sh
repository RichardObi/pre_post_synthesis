#!/bin/sh
################################ Validation ################################

# fid metrics
sh fid_breast_mri.sh

# non-fid metrics
sh metrics_breast_mri.sh