The code was tested on python 3.8 and 3.9

Install the requirements:
```bash
python -m pip install -r requirements.txt
```

### Experiments for Figures 2 and 3
In order to recompute the importance scores run:
```bash
python src/explain_all.py
```
Results will be saved in directory 'explanations'.

In order to redraw plots for Figures 2 and 3 from the newly computed importance scores:
```bash
python src/plot_greedy_all.py
```
The plots will be saved in directory 'plots_greedy'.

### Robustness experiments
```bash
python src/run_robustness.py -m ./datasets/energy/model.pt.zip -d ./datasets/energy -o ./experiments/robustness 
python src/plot_robustness_results.py -i ./explanations/robustness -o ./plots_greedy/robustness  
```
