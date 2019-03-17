%% Examples of benchmarks for different input formats
clear all;close all;clc;
addpath 'bench_fast/benchmarks/'


imgDir = 'BSDS500/data/images/test/';
gtDir = 'BSDS500/data/groundTruth/test/';
inDir = 'kmeans';
outDir = 'eval_kmeans';
mkdir(outDir);
nthresh = 20;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

%plot_eval(outDir)

