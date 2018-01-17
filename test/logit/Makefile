 : Rplot.pdf

Rplot.pdf : bayes-logit-nodal-elbo.csv
	Rscript bayes-logit-nodal.R

bayes-logit-nodal-elbo.csv : 
	julia test-bayes-logit.jl

clean : 
	rm -f *.csv *.out *.pdf