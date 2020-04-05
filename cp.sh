#for finding spatial_distribution of urea about model system for all concentration
echo $1
i=0
while [ $i -ge 0 ]
do
        echo $i
	cp  spatial-density-distribution.py $i'input/'$1 
	cd $i'input/'$1
		python  spatial-density-distribution.py $1"_"$i"murea_eq_npt_recenter_orient.dcd" $1"_"$i"murea"$4".pdb" $2 "plot_dcd" $1"_charmm.psf" $3 "O" "False"
	
	cd ../..
	i=$[$i-1]
done


