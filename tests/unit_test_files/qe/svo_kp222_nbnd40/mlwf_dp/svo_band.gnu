set style data dots
set nokey
set xrange [0: 4.20916]
set yrange [  4.32355 : 19.07852]
set arrow from  1.41665,   4.32355 to  1.41665,  19.07852 nohead
set arrow from  2.23456,   4.32355 to  2.23456,  19.07852 nohead
set arrow from  3.05247,   4.32355 to  3.05247,  19.07852 nohead
set xtics ("R"  0.00000,"G"  1.41665,"X"  2.23456,"M"  3.05247,"G"  4.20916)
 plot "svo_band.dat"
