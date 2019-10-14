#!/usr/bin/env python3
import subprocess
import 


if __name__ == "__main__":

subprocess.Popen("cwm --rdf test.rdf --ntriples > test.nt")

os.
part done
2
3
not done
4-80


for i in $(seq 10 115)
do
    python download.py -s $i
    python audio_preprocessing/generate_features.py -f mfcc -s $i
    directory=".audio/s"
    directory+=$i
    rm -r $directory
done




python download.py -s 2
for i in $(seq 1 287)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 2 -a $i
    directory=".audio/s2/a"
    directory+=$i
    rm -r $directory
done


python download.py -s 3
for i in $(seq 1 201)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 3 -a $i
    directory=".audio/s3/a"
    directory+=$i
    rm -r $directory
done

python download.py -s 4
for i in $(seq 1 177)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 4 -a $i
    directory=".audio/s4/a"
    directory+=$i
    rm -r $directory
done

python download.py -s 5
for i in $(seq 1 121)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 5 -a $i
    directory=".audio/s5/a"
    directory+=$i
    rm -r $directory
done

python download.py -s 6
for i in $(seq 1 166)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 6 -a $i
    directory=".audio/s6/a"
    directory+=$i
    rm -r $directory
done

python download.py -s 7
for i in $(seq 1 207)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 7 -a $i
    directory=".audio/s7/a"
    directory+=$i
    rm -r $directory
done

python download.py -s 8
for i in $(seq 1 76)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 8 -a $i
    directory=".audio/s8/a"
    directory+=$i
    rm -r $directory
done

python download.py -s 9
for i in $(seq 1 130)
do
    python audio_preprocessing/generate_features.py -f mfcc -s 9 -a $i
    directory=".audio/s9/a"
    directory+=$i
    rm -r $directory
done


