Workflow
========

* Create Settings File
* Transform Raw Data
* Run Grid Search
* Visualize Best Model

Note: "Best model" might not actualy be the morst performant model depending on which metric you are considering.
currently this is chosen with a single metric and a known problem is that, if you would like the "best" model by some
metric, the entire grid search would have to be re-run

Create Settings File
--------------------

:mod:`src.new_settings_file.main`

.. automodule:: src.new_settings_file.main
   :members:
   :undoc-members: