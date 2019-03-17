%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;

%% 2.   morphological version for :boundary benchmark for results stored as contour images

imgDir = 'data/images';
gtDir = 'data/groundTruth';
pbDir = 'data/png';
outDir = 'eval/test_bdry_fast';
mkdir(outDir);
nthresh = 99;

tic;
boundaryBench_fast(imgDir, gtDir, pbDir, outDir, nthresh);
toc;


%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

imgDir = 'data/images';
gtDir = 'data/groundTruth';
inDir = 'data/ucm2';
outDir = 'eval/test_all_fast';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

