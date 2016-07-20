set term png
set yrange [0.7:0.78]
set ylabel 'AMS'

set xlabel 'Shrinkage'
set xrange [0.02:0.1]
set output 'Shrinkage.png'
plot 'gesamt' u 1:6

set xlabel 'Tree Depth'
set xrange [3:10]
set output 'Depth.png'
plot 'gesamt' u 2:6

set xlabel 'Number of Trees'
set xrange [200:1000]
set output 'Nt.png'
plot 'gesamt' u 3:6

set xlabel 'nEvents (%)'
set xrange [6:10]
set output 'nEvents.png'
plot 'gesamt' u 4:6

set xlabel 'nCuts'
set xrange [100:600]
set output 'nCuts.png'
plot 'gesamt' u 5:6