cores=8
trials=10

echo 'dist'
for i in `seq 2 $cores`
do
    echo $i
    echo '==='
    for j in `seq 1 $trials`
    do
        mpiexec -n $i ./dist_main.o
    done
done

echo 'dist packed'
for i in `seq 2 $cores`
do
    echo $i
    echo '==='
    for j in `seq 1 $trials`
    do
        mpiexec -n $i ./dist_main.o
    done
done

echo 'serial'
for j in `seq 1 $trials`
do
    ./serial_test.o
done
