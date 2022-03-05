# SANDS
This is an annonymous repository containing code and data necessary to
reproduce the results published in
"Semi-supervised Stance Detection of Tweets Via Distant Network Supervision"
for the proposed method SANDS and compared baselines.

#### Steps to run SANDS
1. Install the required packages mentioned [here](SANDS/requirements.txt).
2. Download and extract the [data](https://drive.google.com/file/d/1kJuNjSGwT3riZFyMsvm28TBbjYY8neER/view?usp=sharing) and place under the
[working directory](SANDS/)
3. Change directory to SANDS/SANDS/codes and run 'python3 run_model.py $dataname $splitsize $numclasses' where _$dataname_ can be either INDIA or USA,
_$splitsize_ among 500, 1000, and, 1500. _$numclasses_ currently support 5 and 7 for USA and INDIA arguments, respectively.
