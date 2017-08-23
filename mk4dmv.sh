#!/bin/bash
read str
#echo $str
mkdir $str"_1"
mkdir $str"_2"
mkdir $str"_3"

for i in 0 1 2
do
	mv $str"/frame"$i* $str"_1"
done
for i in 3 4
do
        mv $str"/frame"$i* $str"_2"
done
for i in 5 6
do
        mv $str"/frame"$i* $str"_3"
done

mv $str $str"_4"
