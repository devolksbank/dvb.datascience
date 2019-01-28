import abc
import json
import logging
import re
import pickle
import pathlib
import shutil
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import ipywidgets
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
from pyensae.graphhelper import draw_diagram
import dask.dataframe as dd

from .pipe_base import Data, Params, PipeBase

logger = logging.getLogger(__name__)

default_dataframe_engine = 'pandas'

class Status(Enum):
    NOT_STARTED = 1
    PROCESSING = 2
    FINISHED = 3


class OutputStoreBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save_pipe_output(
        self, pipe_name: str, transform_nr: int, output: Dict[str, Any]
    ) -> None:
        """save the pipe output"""

    @abc.abstractmethod
    def load_pipe_output(self, pipe_name: str, transform_nr: int) -> Dict[str, Any]:
        """load the pipe outputs, returns None when the output does not exists"""

    @abc.abstractmethod
    def contains_pipe(self, pipe_name: str, transform_nr: int) -> bool:
        """returns True when the pipe with pipe_name has stored his output.
        For some implementations this is more efficient as checking whether load_pipe_output is not None
        """
    @abc.abstractmethod
    def clear(self):
        """empty the storage"""


class OutputStoreMemory(OutputStoreBase):
    """
    Store the outputs of the pipes in memory.

    >>> os = OutputStoreMemory()
    >>> Pipeline(output_store=os)
    """

    def __init__(self):
        self.clear()

    def save_pipe_output(
        self, pipe_name: str, transform_nr: int, output: Dict[str, Any]
    ) -> None:
        self.outputs[(pipe_name, transform_nr)] = output

    def load_pipe_output(self, pipe_name: str, transform_nr: int) -> Dict[str, Any]:
        return self.outputs[(pipe_name, transform_nr)]

    def contains_pipe(self, pipe_name: str, transform_nr: int) -> bool:
        return (pipe_name, transform_nr) in self.outputs

    def clear(self):
        self.outputs: Dict[Tuple[str, int], Dict[str, Any]] = {}


class OutputStoreFile(OutputStoreBase):
    """ Store the output of the pipes in files.

    >>> os = OutputStoreFile()
    >>> Pipeline(output_store=os)

    Note: the directory with the outputs will be cleared when the pipeline is
    initialized.
    """
    def __init__(
        self,
        directory="transform_output",
        file_pattern="{pipe_name}_{transform_nr}.pkl",
    ):
        """
        Initalize the output store with an optional specified directory and pattern of the filenames
        """
        self.directory = pathlib.Path(directory)
        self.file_pattern = file_pattern

        self.clear()

    def save_pipe_output(
        self, pipe_name: str, transform_nr: int, output: Dict[str, Any]
    ) -> None:
        filename = self.file_pattern.format(
            pipe_name=pipe_name, transform_nr=transform_nr
        )
        with open(self.directory / filename, "wb") as f:
            pickle.dump(output, f)

    def load_pipe_output(self, pipe_name: str, transform_nr: int) -> Dict[str, Any]:
        filename = self.file_pattern.format(
            pipe_name=pipe_name, transform_nr=transform_nr
        )
        if not (self.directory / filename).exists():
            raise KeyError(
                "No pipe %s with transform %s is available" % (pipe_name, transform_nr)
            )

        with open(self.directory / filename, "rb") as f:
            return pickle.load(f)

    def contains_pipe(self, pipe_name: str, transform_nr: int) -> bool:
        filename = self.file_pattern.format(
            pipe_name=pipe_name, transform_nr=transform_nr
        )
        return (self.directory / filename).exists()

    def clear(self):
        if self.directory.exists():
            shutil.rmtree(self.directory)

        self.directory.mkdir(parents=True, exist_ok=True)

class Pipeline:

    """
    A connector specifies which Pipe (identified by its name) and which output from that
    Pipe (identified by the key of the output) will be input to a Pipe (identified by its name)
    and which input for that Pipe (identified by its key)

    Example

    >>> pipeline = Pipeline()
    >>> pipeline.addPipe('read', ReadCSV())
    >>> pipeline.addPipe('impute', Impute(), [("read", "df", "df")])

    >>> pipeline.fit()

    >>> pipeline.transform()

    The constructor of Pipeline also accepts a `output_store` argument, which
    must be an instance of OutputStoreBase. At default, an OutputStoreMemory will be used.
    """

    def __init__(
        self,
        output_store: Optional[OutputStoreBase] = None,
        dataframe_engine: str = None,
    ) -> None:
        logger.info("Initiate pipeline")
        if output_store is None:
            output_store = OutputStoreMemory()
        self.output_store: OutputStoreBase = output_store

        if dataframe_engine is None:
            dataframe_engine = default_dataframe_engine
        if dataframe_engine not in ('pandas', 'dask'):
            raise ValueError('dataframe engine %s is not supported' % dataframe_engine)
        self.dataframe_engine = dataframe_engine

        self.pipes: Dict[str, PipeBase] = {}
        self.input_connectors: Dict[Optional[str], List] = defaultdict(list)
        """A mapping between an input pipe name and related connection tuples (output_name, output_key, input_name, input_key)"""
        self.output_connectors: Dict[Optional[str], List]  = defaultdict(list)
        """A mapping between an output pipe name and related connection tuples (output_name, output_key, input_name, input_key)"""

        # Input containing nodes/edge definitions for blockdiag
        self.diagram_definition = ""
        self.diagram_notes = ""

        self.reset_fit()

    def reset_fit(self):
        self.current_transform_nr = -1
        self.transform_status = {}
        self.output_store.clear()

    @staticmethod
    def is_valid_name(name):
        name_regex = re.compile(r"[^a-zA-Z0-9_]")
        return not bool(name_regex.search(name))

    def _calculate_note_dimensions(self, node_name: str) -> Tuple[int, int]:
        """
        Scale the notes box dimensions to make the text fit. Note that all values in this method have been set experimentally
        """

        # Start with default width/height
        width = 128
        height = 40

        min_comment_length_for_scaling = 25
        if len(node_name) > min_comment_length_for_scaling:
            logger.info(
                "Comment length[%s] is more than scaling threshold[%s]",
                len(node_name),
                min_comment_length_for_scaling,
            )

            width += 2 * len(node_name)
            height *= int(1 + (len(node_name) / (min_comment_length_for_scaling * 6)))

        logger.info("Returning width[%s], height[%s]", int(width), int(height))
        return int(width), int(height)

    def addPipe(
        self,
        name: str,
        pipe: PipeBase,
        inputs: List[Tuple[Union[str, PipeBase], str, str]] = None,
        comment: str = None,
    ) -> "Pipeline":
        """
        Add a pipe `pipe` to the pipeline with the given name. Optionally add the input connectors by
        adding them to `inputs`. `inputs` is a list of the inputs whith for each input a tuple with
        (output_pipe, output_key, input_key).
        """
        logger.info("Adding pipe named: %s", name)

        if not self.is_valid_name(name):
            raise ValueError("%s is no valid name" % name)
        if len(self.transform_status) > 0:
            self.reset_fit()
            display(
                HTML(
                    "<p><i>Note: resetting fit, because a fitted pipeline can not be changed.</i></p>"
                )
            )
        if name in self.pipes:
            raise ValueError(
                "A pipe with name '" + name + "' is already present in ths pipeline"
            )

        pipe.name = name
        pipe.setPipeline(self)
        self.pipes[name] = pipe

        if inputs:
            logger.info("Processing inputs for pipe")
            fixed_inputs = [
                (t[0].name if isinstance(t[0], PipeBase) else t[0], t[1], t[2])
                for t in inputs
            ]

            for currentInput in fixed_inputs:
                if not isinstance(currentInput[0], str):
                    raise ValueError(
                        "first item of input must be a string or a PipeBase instance, however it is '%s'"
                        % currentInput[0]
                    )
                if currentInput[0] not in self.pipes:
                    raise ValueError(
                        "A pipe with name '%s' is not present in the pipeline"
                        % currentInput[0]
                    )
                if currentInput[1] not in self.pipes[currentInput[0]].output_keys:
                    raise ValueError(
                        "The pipe with name '%s' has not '%s' as an output key"
                        % (currentInput[0], currentInput[1])
                    )
                if currentInput[2] not in self.pipes[name].input_keys:
                    raise ValueError(
                        "The pipe with name '%s' has not '%s' as an input key"
                        % (name, currentInput[2])
                    )
                self._connect(currentInput[0], currentInput[1], name, currentInput[2])
        else:
            self._connect(None, None, name, None)

        if comment is not None:
            logger.info("Attaching comment step for visualization")

            width, height = self._calculate_note_dimensions(node_name=comment)

            # Add the group for making the pair in landscape mode
            # Note: make use of the bug in the layout orientation placement as described here:
            #       https://github.com/blockdiag/blockdiag/issues/80
            self.diagram_notes += " group { orientation = portrait; group { \n \n "

            # Set the style of the notes box
            self.diagram_notes += (
                name
                + "_comment [shape = flowchart.terminator, color = '#ffff00', textcolor='#000000', label = '"
                + comment
                + "', width="
                + "{0:g}".format(float(width))
                + ", height="
                + "{0:g}".format(float(height))
                + "]\n"
            )

            # Make the transition to the notes box
            self.diagram_notes += name + " -> " + name + "_comment [style = dashed ]\n "

            # Close the groups
            self.diagram_notes += "}\n " + "}\n "

        return self

    def _connect(
        self,
        output_name: Optional[str],
        output_key: Optional[str],
        input_name: Optional[str],
        input_key: Optional[str],
    ):
        c = (output_name, output_key, input_name, input_key)

        if output_name is not None:
            self.output_connectors[output_name].append(c)
            self.input_connectors[input_name].append(c)
        else:
            output_name = "Start"
            output_key = ""
            input_key = ""

        # Note: 'special' characters have already been removed in processing the names
        logger.info(
            "Adding diagram definition for output_name[%s], output_key[%s], input_name[%s], input_key[%s]",
            output_name,
            output_key,
            input_name,
            input_key,
        )

        # Use the more informational name when available
        label: str = output_key if output_key is not None else ""
        if input_key != output_key and input_key is not None:
            label += "-->" + input_key

        # Calculate node dimensions
        width, height = self._calculate_note_dimensions(node_name=label or "")

        # Append the node
        self.diagram_definition += "{} [ width={:g}, height={:g} ]\n".format(
            input_name, float(width), float(height)
        )

        # Append the new transition in the data for visualization
        self.diagram_definition += '{} -> {} [label= "{}" ]\n'.format(
            output_name, input_name, label
        )

        # Highlight the first node
        if output_name == "Start":
            self.diagram_definition += "Start [color = lightblue]\n "

    def draw_design(self):
        """
        Returns an image with all pipes and connectors.
        """
        display("Drawing diagram using blockdiag")

        block_diag_print = (
            'blockdiag {default_shape = roundedbox; default_group_color = "#FFFFFF"; orientation ="portrait"\n  '
            + self.diagram_definition
            + self.diagram_notes
            + "\n}\n "
        )
        logger.info("Using definition for blockdiag: %s", block_diag_print)
        img2 = draw_diagram(block_diag_print)
        display(img2)

    def get_processable_pipes(self) -> List[PipeBase]:
        """
        get the pipes which are processable give the status of the pipeline
        """
        processable_pipes = []
        for pipe_name, pipe in self.pipes.items():
            # check if pipe is already processed
            if self.output_store.contains_pipe(
                pipe_name, self.current_transform_nr
            ):
                continue

            if len(pipe.input_keys) == 0:
                # no input is needed, so can be processes always
                processable_pipes.append(pipe)
                continue

            # check if input is present. ie check if all needed outputs for the input are present
            if not all(
                [
                    self.output_store.contains_pipe(
                        output_name, self.current_transform_nr
                    )
                    for output_name, _, _, _ in self.input_connectors[pipe_name]
                ]
            ):
                # needed pipe is not processed yet
                continue

            processable_pipes.append(pipe)

        return processable_pipes

    def get_pipe(self, name) -> Optional[PipeBase]:
        return self.pipes.get(name)

    def get_pipe_input(self, name) -> Optional[Dict]:
        """
        Get the input for the pipe with `name` from the transformed outputs.
        Returns a dict with all data when all data for the pipe are collectable.
        Returns None when not all data is present yet
        """
        r = {}
        for output_name, output_key, _, input_key in self.input_connectors[name]:
            if not self.output_store.contains_pipe(
                output_name, self.current_transform_nr
            ):
                return None
            value = self.output_store.load_pipe_output(
                output_name, self.current_transform_nr
            )
            if output_key not in value:
                return None
            r[input_key] = value[output_key]

        return r

    @classmethod
    def _compute_when_needed(cls, data: Data):
        """
        Data can contain lazy DataFrames. When a df is lazy, that one is computed.
        """
        d = {}

        for key, df in data.items():
            if type(df) in (dd.DataFrame, dd.Series):
                df = df.compute()
            d[key] = df

        return d

    def get_pipe_output(self, name: str, transform_nr: int = None) -> Dict:
        """
        Get the output of the pipe with `name` and the given `transform_nr` (which default to None
        which will selects the last one). When no output
        is present, an empty dict is returned
        """
        if transform_nr is None:
            transform_nr = self.current_transform_nr

        return self._compute_when_needed(
                self.output_store.load_pipe_output(name, self.current_transform_nr)
            )

    @staticmethod
    def get_params(params: Dict, key: str, metadata: Dict = None) -> Dict:
        """
        Get a dict with the contents of params only relevant for the pipe with the given key as name. Besides
        that, also the params['default'] and metadata will be added.
        """
        d = {}
        if metadata:
            d.update({"metadata": metadata})
        d.update(params.get("default", {}))
        d.update(params.get(key, {}))
        return d

    def end(self):
        """
        When all fit and transforms are finished, end the pipeline, so some clean up can be done.
        At this moment, that is mainly needed to close plots, so they won't be shown twice in the notebook
        """
        plt.close("all")

    def fit_transform(
        self,
        data: Optional[Data] = None,
        transform_params: Optional[Params] = None,
        fit_params: Optional[Params] = None,
        name: str = "fit",
        close_plt: bool = False,
    ) -> None:
        """
        Train all pipes in the pipeline and run the transform for the first time
        """
        self.reset_fit()

        self.transform(
            data=data,
            transform_params=transform_params,
            fit_params=fit_params,
            fit=True,
            name=name,
            close_plt=close_plt,
        )

    def _is_pipe_input_for_another_pipe(self, pipe: PipeBase):
        """
        Returns True when the provided pipe is an input for another pipe in the pipeline
        """
        for _, _, input_name, _ in self.output_connectors[pipe.name]:
            if input_name is not None:
                return True
        return False

    def fit_transform_try(self, *args, **kwargs):
        try:
            self.fit_transform(*args, **kwargs)
        except Exception:
            import traceback

            traceback.print_exc()

    def transform_try(self, *args, **kwargs):
        try:
            self.transform(*args, **kwargs)
        except Exception:
            import traceback

            traceback.print_exc()

    def transform(
        self,
        data: Optional[Data] = None,
        transform_params: Optional[Params] = None,
        fit_params: Optional[Params] = None,
        fit: bool = False,
        name: Optional[str] = None,
        close_plt: bool = False,
    ):
        """
        When transform_params or fit_params contain a key 'default', that params will
        be given to all pipes, unless it is overridden by a specific value for that pipe in
        transform_params or fit_params. The default can be useful for params which are
        needed in a lot of pipes.
        """
        self.draw_design()

        self.current_transform_nr += 1

        if fit and self.current_transform_nr > 0:
            raise ValueError("Cannot fit, because a fit has already been processed.")

        if name is None:
            if fit:
                name = "fit"
            else:
                name = "transform %s" % self.current_transform_nr

        metadata = {"name": name, "nr": self.current_transform_nr, "fit": fit}

        display(HTML("<h1>Transform %s</h1>" % name))

        self.transform_status[self.current_transform_nr] = Status.PROCESSING

        # when present, the input argument contains the input for the pipeline which will be stored
        # as a pipe with '' as name, so other pipes can get that as input
        if data:
            self.output_store.save_pipe_output("", self.current_transform_nr, data)

        if transform_params is None:
            transform_params = {}
        if fit_params is None:
            fit_params = {}

        progress_label = ipywidgets.HTML()
        progress_bar = ipywidgets.FloatProgress(
            min=0, max=len(self.pipes), description="Progress:"
        )
        progress_box = ipywidgets.HBox(children=[progress_bar, progress_label])
        display(progress_box)

        while True:
            pipes = self.get_processable_pipes()
            logger.info("collected pipes for next run: %s", pipes)
            if len(pipes) == 0:
                break

            for pipe in pipes:
                pipe_input = self.get_pipe_input(pipe.name)
                if pipe_input is None:
                    continue
                progress_label.value = pipe.name
                pipe_transform_params = self.get_params(
                    transform_params, pipe.name, metadata
                )
                pipe_fit_params = self.get_params(fit_params, pipe.name, metadata)

                if fit:
                    # fit_and_transform
                    logger.info("fit pipe %s", pipe.name)
                    output = pipe.fit_transform(
                        data=pipe_input,
                        fit_params=pipe_fit_params,
                        transform_params=pipe_transform_params,
                    )
                else:
                    # only transform
                    if pipe_transform_params.get("skip", False):

                        if pipe.output_keys is None:
                            # No data to output, so skipping is fine!
                            output = {key: None for key in pipe.output_keys}
                        elif not self._is_pipe_input_for_another_pipe(pipe):
                            # No other pipes are attached to this pipe, so skipping it is fine also!
                            output = {key: None for key in pipe.output_keys}
                        elif pipe.output_keys != pipe.input_keys:
                            # skip this pipe; all input will be directed to the output
                            # so this works only for pipes with has the same input and output
                            raise ValueError(
                                "A pipe can only be skipped when the input_keys and the output_keys are "
                                "the same. Pipe %s has different keys for input and output"
                                % pipe.name
                            )
                        else:
                            output = pipe_input
                    else:
                        # (not skipping the execution of this pipe)
                        logger.info("transform pipe %s", pipe.name)
                        output = pipe.transform(
                            data=pipe_input, params=pipe_transform_params
                        )

                # check if all expected is present
                if not isinstance(output, dict):
                    raise ValueError(
                        "Pipe %s didn't return a dict (actual: %s)"
                        % (pipe.name, repr(output))
                    )

                if not all([(key in output) for key in pipe.output_keys]):
                    raise ValueError(
                        "Pipe '%s' does not contain all expected output (present: '%s', expected: '%s')"
                        % (pipe.name, list(output.keys()), pipe.output_keys)
                    )

                self.output_store.save_pipe_output(
                    pipe.name, self.current_transform_nr, output
                )
                progress_bar.value += 1
        progress_label.value = "finished"
        progress_bar.bar_style = "success"
        self.transform_status[self.current_transform_nr] = Status.FINISHED

        if close_plt:
            self.end()

        return

    def load(self, file_path: str) -> None:
        """
        Load the fitted parameters from the file in `file_path` and load them in all Pipes.
        """
        with open(file_path) as f:
            state = json.load(f)
            for pipe_name, pipe in self.pipes.items():
                pipe.load(state.get(pipe_name, {}))

    def save(self, file_path: str) -> None:
        """
        Save the fitted parameters from alle Pipes to the file in `file_path`.
        """
        state = {}
        for pipe_name, pipe in self.pipes.items():
            state[pipe_name] = pipe.save()

        with open(file_path, "w") as f:
            json.dump(state, f)
