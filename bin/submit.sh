#!/bin/bash

notebook=$1 && shift 
outdir=scratch/$1 && shift


jnotebook=$(echo $outdir | sed 's%logs_%%;').ipynb
jobname=$(echo $jnotebook | sed 's%.ipynb%%;')

mkdir $outdir

echo $jobname > $outdir/jobname
cp -p $notebook $jnotebook

sbatch -J $jobname -o $outdir/slurm.log my_job.sh $jnotebook --Parameters.batch=True --Parameters.monitor_dir=$outdir $@

