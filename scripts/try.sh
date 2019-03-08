export DATA_DIR="$1"
export SUPPORT_DIR="$2"

echo "DATA_DIR "$DATA_DIR
echo "SUPPORT_DIR "$SUPPORT_DIR

for i in $(seq 1 2)
do 
    echo $i 
    echo moving $DATA_DIR/*.csv to 
    mv $DATA_DIR/*.csv $SUPPORT_DIR/weight_0pt$i/
done