#!/bin/bash

# Running the script takes about 10 minutes

mkdir data
pushd data

# BrightKite dataset (for caching and ice cream MTS)

# download and unzip
wget http://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz
gzip -d loc-brightkite_totalCheckins.txt.gz
# extract ids of users with 2100 checkins (the maximum number in the dataset)
cut -f1 loc-brightkite_totalCheckins.txt | uniq -c | sed 's/^ *//' | grep '^2100' | cut -d' ' -f2 > brightkite_topUsers.txt
# extract data of those users, and prepare temporary caching datasets for them
for u in `cat brightkite_topUsers.txt`
do
  grep -P '^'$u'\t' loc-brightkite_totalCheckins.txt > full_bk$u.txt
  cut -f 5 full_bk$u.txt > bk$u.txt
done
# compute opt for top users, and select those who require at least 50 cache misses
../main.py opt 10 bk*.txt > brightkite_opt.txt
paste brightkite_topUsers.txt brightkite_opt.txt | awk '$2 >= 50 {print $1}' | head -n 100 > brightkite_selectedUsers.txt
# remove temporary caching datasets
rm bk*.txt
# prepare final datasets for caching and ice cream MTS
for u in `cat brightkite_selectedUsers.txt`
do
  # caching: just select the location hash
  cut -f 5 full_bk$u.txt > bk$u.txt
  # ice cream MTS: compute median latitutde, everything above is vanilla, below is chocolate
  median=`cut -f 3 full_bk$u.txt | sort | head -n 1050 | tail -n 1`
  awk '{print ($3>'$median')?"V":"C"}' full_bk$u.txt > ic$u.txt
done
# remove temporary files
rm full_bk*.txt
rm loc-brightkite_totalCheckins.txt
rm brightkite_topUsers.txt
rm brightkite_selectedUsers.txt
rm brightkite_opt.txt

# Citibike dataset (for caching only)

# for each month of 2017 take first 25k rides, and treat start location ids as page numbers
for m in 01 02 03 04 05 06 07 08 09 10 11 12
do
  fname=2017$m-citibike-tripdata.csv
  wget https://s3.amazonaws.com/tripdata/$fname.zip
  unzip $fname.zip
  rm $fname.zip
  cut -d, -f4 $fname | tail -n+2 | head -n 25000 > citi$m.txt
  rm $fname
done

popd
