ett_mul="HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"
ett_mon="OT,"
#********************************************************
exchange_mul="0,1,2,3,4,5,6"
exchange_mon="OT,"
#********************************************************
elec_mul="0"
for i in {1..319}; do
    elec_mul="$elec_mul,$i"
done
elec_mon="OT,"
#********************************************************
traffic_mul="0"
for i in {1..860}; do
    traffic_mul="$traffic_mul,$i"
done
traffic_mon="OT,"
#********************************************************
pems_mul="0"
for i in {0..357}; do
    pems_mul="$pems_mul,$i"
done
pems_mon="357,"
#********************************************************
weather_mul="p,T,Tpot,Tdew,rh,VPmax,VPact,VPdef,sh,H2OC,rho,wv,max.wv,wd,rain,raining,SWDR,PAR,max.PAR,Tlog,OT"
weather_mon="OT,"
#********************************************************
solar_mul="0"
for i in {1..136}; do
    solar_mul="$solar_mul,$i"
done
solar_mon="136,"


mono_multi_names=("monomono" "multimono" "multimulti")

run() {
    for pred_len in {96,192,336,720}; do
        exp=$name"_"$pred_len"_"$mono_multi
        echo
        echo "*****************************************************"
        echo ">> Running Expriment: $exp"
        echo "*****************************************************"
        echo
        time python3 -u experiments/run_exp.py \
                        --config configs/config.yml \
                        --set run.name=$exp \
                        --set data.csv.paths=$data_file \
                        --set data.csv.feature_cols=$in_params \
                        --set data.csv.target_cols=$out_params \
                        --set data.csv.has_header=$has_head \
                        --set data.csv.use_col_indices=$use_indx \
                        --set task.d_in=$(( $(grep -o ',' <<<"$in_params" | wc -l) + 1 )) \
                        --set task.d_out=$(( $(grep -o ',' <<<"$out_params" | wc -l) + 1 )) \
                        --set task.pred_length=$pred_len \
                        --set data.csv.date_col=$date
        echo
        echo "#####################################################"
        echo "<< Done With Expriment: $exp"
        echo "#####################################################"
        echo
    done
}

run4data() {
    for i in {0..2}; do
        in_params="${ett_in_params[i]}"
        out_params="${ett_out_params[i]}"
        mono_multi="${mono_multi_names[i]}"
        run
    done
}

DATA_DIR="/Users/a.e./Dropbox/evots/data/iTransformer_datasets"


#############################
# *** DATA FILES: Weather ***
#############################
has_head=true
use_indx=false
date="0"

ett_in_params=($weather_mon $weather_mul $weather_mul)
ett_out_params=($weather_mon $weather_mon $weather_mul)

#-----------------------------
data_file=$DATA_DIR/weather/weather.csv
name=weather.csv
run4data
#-----------------------------

#########################
# *** DATA FILES: ETT ***
#########################
has_head=true
use_indx=false
date="0"

ett_in_params=($ett_mon $ett_mul $ett_mul)
ett_out_params=($ett_mon $ett_mon $ett_mul)

#-----------------------------
data_file=$DATA_DIR/ETT-small/ETTh1.csv
name=etth1
run4data
#-----------------------------

#-----------------------------
data_file=$DATA_DIR/ETT-small/ETTh2.csv
name=etth2
run4data
#-----------------------------

#-----------------------------
data_file=$DATA_DIR/ETT-small/ETTm1.csv
name=ettm1
run4data
#-----------------------------

#-----------------------------
data_file=$DATA_DIR/ETT-small/ETTm2.csv
name=ettm2
run4data
#-----------------------------


#################################
# *** DATA FILES: Electricity ***
#################################
has_head=true
use_indx=false
date="0"

ett_in_params=($elec_mon $elec_mul $elec_mul)
ett_out_params=($elec_mon $elec_mon $elec_mul)

#-----------------------------
data_file=$DATA_DIR/electricity/electricity.csv
name=electricity
run4data
#-----------------------------


##############################
# *** DATA FILES: Exchange ***
##############################
has_head=true
use_indx=false
date="0"

ett_in_params=($exchange_mon $exchange_mul $exchange_mul)
ett_out_params=($exchange_mon $exchange_mon $exchange_mul)

#-----------------------------
data_file=$DATA_DIR/exchange_rate/exchange_rate.csv
name=exchange
run4data
#-----------------------------


#############################
# *** DATA FILES: Traffic ***
#############################
has_head=true
use_indx=false
date="0"

ett_in_params=($traffic_mon $traffic_mul $traffic_mul)
ett_out_params=($traffic_mon $traffic_mon $traffic_mul)

#-----------------------------
data_file=$DATA_DIR/traffic/traffic.csv
name=traffic
run4data
#-----------------------------


###########################
# *** DATA FILES: Solar ***
###########################
has_head=false
use_indx=true
date="null"

ett_in_params=($solar_mon $solar_mul $solar_mul)
ett_out_params=($solar_mon $solar_mon $solar_mul)

#-----------------------------
data_file=$DATA_DIR/Solar/solar_AL.txt
name=solar
run4data
#-----------------------------

##########################
# *** DATA FILES: PEMS ***
##########################
has_head=false
use_indx=true
date="null"

ett_in_params=($pems_mon $pems_mul $pems_mul)
ett_out_params=($pems_mon $pems_mon $pems_mul)

#-----------------------------
data_file=$DATA_DIR/pems/PEMS03.npz
name=pems03
run4data
#-----------------------------

#-----------------------------
data_file=$DATA_DIR/pems/PEMS04.npz
name=pems04
run4data
#-----------------------------


#-----------------------------
data_file=$DATA_DIR/pems/PEMS07.npz
name=pems07
run4data
#-----------------------------

#-----------------------------
data_file=$DATA_DIR/pems/PEMS08.npz
name=pems08
run4data
#-----------------------------
