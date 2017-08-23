#!/bin/bash
read str
#echo $str
mkdir $str"_result"

mv *pickle $str"_result"

zip -r $str"_result.zip" $str"_result"
