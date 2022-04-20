
# Project structure

At high level this project complies with the following scheme where the big folder like element at the begining is the project folder, the following folder like elements are packages and the simple rounded square are modules:


![High Level Project Structure](/outputs/README_images/platges_framework_with_global_1247_1256.png "High Level Project Structure")


The framework package contain tools useful for this and other projects, it could be installed, added on the pythonpath or leave it on the folder.

The platges_utils package uses the framework package to make functions that build each part of the pyTorch optimizable experiments for this specific project or (the platges_utils package) contain other not executable modules; this reduce the line count on the final script while mantaining the flow as much as possible. If the framework package is not installed or added on the pythonpath, the package will try to add it without guarantees of success.

TODO: The other packages are more or less easier and I may document them later.

---

<br>

# Deep Learning experiments

The experiments should follow the next structure:

    # IMPORTS


    # MACHINE RELATED CONSTANTS (if needed)

    # EXPERIMENT DEFAULT METADATA (SimpleNamespace)

    # MODEL DEFAULT HYPERPARAMETERS (SimpleNamespace)


    # SimpleNamespace-RELATED FUNCTIONS


    ####################################

    # SMALL FUNCTIONS/CLASSES

    # MAIN_FUNCTION
        # GET DATA-RELATED PARAMETERS
        # BUILD DATALOADERS
        # BUILD MODEL
        # BUILD LOSS
        # BUILD OPTIM
        # BUILD LOCAL LOGGER with METRICS
        # BUILD OUTPUT LOGGERS
        # TRAINING RUTINE

    ####################################


    # MAIN
        # OVERWRITE DEFAULTS (both SimpleNamespace) FROM DOCOPTS
        # CALL MAIN_FUNCTION (with the SimpleNamespaces)

Each element from the the previous pseudocode should use as little lines as possible in order to easily give a clear view of what is happening.

---

# K-Folds hacks

When training with k-folds: the datasets/dataloader, the model, the optim/loss and the loggers change on each fold of the training rutine; the dataloader parameter will return an iterator of dataloaders (a zip of iterators does not expand them) and similar with the model.

The loss may requiere the model as construction parameters to have a regularization part, this will be added at the constructor although it may not be needed (through lambda function, decorator, etc); the kfolds needs to reinitialize the loss each time (some losses have memory).

The optim requiere the model as construction parameter; the kfolds needs to reinitialize the optimizer each time.

In order to log the kfolds experiment, a new logger will manage the kfolds data and build the needed local loggers.

---

# Rutines

Although training rutines can be reusable along projects, it is recomendable to know what the code do; with this idea in mind, the rutines will be defined on the project specific package "platges_utils"; and, on the framework package there will be a utils subpackage where subrutines like scale process will be available.

---
