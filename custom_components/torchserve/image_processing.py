"""
Component that will perform inference via torchserve.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/image_processing.deepstack_object
"""
from collections import namedtuple
import io
import logging
import time
import re
from pathlib import Path

from PIL import Image, ImageDraw

import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.util.pil import draw_box

import grpc
import requests

import custom_components.torchserve.inference_pb2_grpc as inference_pb2_grpc
import custom_components.torchserve.management_pb2_grpc as management_pb2_grpc

from homeassistant.components.image_processing import (
    ATTR_CONFIDENCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_LAST_TRIP_TIME,
    CONF_IP_ADDRESS,
    CONF_COUNT,
)
from homeassistant.core import split_entity_id

_LOGGER = logging.getLogger(__name__)

CONF_TARGETS = "targets"
CONF_MODELS = "models"
CONF_TIMEOUT = "timeout"
CONF_SAVE_FILE_FOLDER = "save_file_folder"
CONF_SAVE_TIMESTAMPTED_FILE = "save_timestamped_file"
CONF_SAVE_CROPPED_FILE = "save_timestamped_crops"
CONF_SAVE_LABEL_DATA = "save_label_data"
CONF_SAVE_LATEST = "save_latest"
CONF_SAVE_BLANKS = "save_timestamped_blanks"
CONF_SHOW_BOXES = "show_boxes"
CONF_PORT_MGMT_GRPC = "port_mgmt_grpc"
CONF_PORT_MGMT_REST = "port_mgmt_rest"
CONF_PORT_INFER_GRPC = "port_infer_grpc"
CONF_PORT_INFER_REST = "port_infer_rest"
CONF_FIRE_EVENTS = "fire_events"

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
DEFAULT_TARGETS = []
DEFAULT_TIMEOUT = 10
DEFAULT_PORT_MGMT_GRPC = 7071
DEFAULT_PORT_MGMT_REST = 8081
DEFAULT_PORT_INFER_REST = 7070
DEFAULT_PORT_INFER_GRPC = 8081

DATA_TORCHSERVE = "data_torchserve"
DATA_MODEL = "model"
DATA_BOX = "bounding_box"
DATA_BOX_AREA = "box_area"
DATA_CENTROID = "centroid"
DATA_PREDICTION_TYPE = "prediction_type"
DATA_PREDICTION_TYPE_OBJECT = "object"
DATA_PREDICTION_TYPE_CLASS = "class"

# rgb(red, green, blue)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)


PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_IP_ADDRESS): cv.string,
        vol.Required(CONF_PORT_MGMT_GRPC, default=DEFAULT_PORT_MGMT_GRPC): cv.port,
        vol.Required(CONF_PORT_MGMT_REST, default=DEFAULT_PORT_MGMT_REST): cv.port,
        vol.Required(CONF_PORT_INFER_GRPC, default=DEFAULT_PORT_INFER_GRPC): cv.port,
        vol.Required(CONF_PORT_INFER_REST, default=DEFAULT_PORT_INFER_REST): cv.port,
        vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int,
        vol.Optional(CONF_TARGETS, default=DEFAULT_TARGETS): vol.All(
            cv.ensure_list, [cv.string]
        ),
        vol.Optional(CONF_MODELS): vol.All(
            cv.ensure_list, [cv.string]
        ),
        vol.Optional(CONF_SAVE_FILE_FOLDER): cv.isdir,
        vol.Optional(CONF_SAVE_TIMESTAMPTED_FILE, default=False): cv.boolean,
        vol.Optional(CONF_SAVE_CROPPED_FILE, default=False): cv.boolean,
        vol.Optional(CONF_SAVE_LABEL_DATA, default=False): cv.boolean,
        vol.Optional(CONF_SAVE_BLANKS, default=False): cv.boolean,
        vol.Optional(CONF_SAVE_LATEST, default=True): cv.boolean,
        vol.Optional(CONF_SHOW_BOXES, default=True): cv.boolean,
        vol.Optional(CONF_FIRE_EVENTS, default=True): cv.boolean,
    }
)

Box = namedtuple("Box", "y_min x_min y_max x_max")
Point = namedtuple("Point", "y x")


def get_valid_filename(name: str) -> str:
    """Return validf filename from HA entity name."""
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def get_objects(predictions: list, model: str, img_width: int, img_height: int):
    """Return objects with formatting and extra info."""
    objects = []
    decimal_places = 3
    if isinstance(predictions, dict):
        for key in predictions:
            confidence = predictions[key]
            box = {
                "height": 1,
                "width": 1,
                "y_min": 0,
                "x_min": 0,
                "y_max": 1,
                "x_max": 1,
            }
            box_area = round(box["height"] * box["width"], decimal_places)
            centroid = {
                "x": round(box["x_min"] + (box["width"] / 2), decimal_places),
                "y": round(box["y_min"] + (box["height"] / 2), decimal_places),
            }
            objects.append(
                {
                    DATA_BOX: box,
                    DATA_BOX_AREA: box_area,
                    DATA_CENTROID: centroid,
                    CONF_NAME: key,
                    ATTR_CONFIDENCE: confidence * 100,
                    DATA_MODEL: model,
                    DATA_PREDICTION_TYPE: DATA_PREDICTION_TYPE_CLASS,
                }
            )
    else:
        for pred in predictions:
            if isinstance(pred, str):  # this is image class not object detection so no objects
                return objects
            name = list(pred.keys())[0]

            box_width = pred[name][2] - pred[name][0]
            box_height = pred[name][3] - pred[name][1]
            box = {
                "height": round(box_height / img_height, decimal_places),
                "width": round(box_width / img_width, decimal_places),
                "y_min": round(pred[name][1] / img_height, decimal_places),
                "x_min": round(pred[name][0] / img_width, decimal_places),
                "y_max": round(pred[name][3] / img_height, decimal_places),
                "x_max": round(pred[name][2] / img_width, decimal_places),
            }
            box_area = round(box["height"] * box["width"], decimal_places)
            centroid = {
                "x": round(box["x_min"] + (box["width"] / 2), decimal_places),
                "y": round(box["y_min"] + (box["height"] / 2), decimal_places),
            }
            confidence = round(pred['score'] * 100, decimal_places)

            objects.append(
                {
                    DATA_BOX: box,
                    DATA_BOX_AREA: box_area,
                    DATA_CENTROID: centroid,
                    CONF_NAME: name,
                    DATA_MODEL: model,
                    ATTR_CONFIDENCE: confidence,
                    DATA_PREDICTION_TYPE: DATA_PREDICTION_TYPE_OBJECT,
                }
            )
    return objects


def infer_via_rest(host, port, model_name, model_input):
    """Run inference via REST."""
    url = f"http://{host}:{port}/predictions/{model_name}"
    prediction = requests.post(url, data=model_input).text
    return prediction


def get_model_list_via_rest(host, port):
    """List available models."""
    url = f"http://{host}:{port}/models"
    models = requests.get(url).text
    return models


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up the classifier."""
    save_file_folder = config.get(CONF_SAVE_FILE_FOLDER)
    if save_file_folder:
        save_file_folder = Path(save_file_folder)

    if DATA_TORCHSERVE not in hass.data:
        hass.data[DATA_TORCHSERVE] = []

    targets = [t.lower() for t in config[CONF_TARGETS]]  # ensure lower case
    models = [t for t in config[CONF_MODELS]]
    entities = []
    for camera in config[CONF_SOURCE]:
        object_entity = ObjectClassifyEntity(
            config.get(CONF_IP_ADDRESS),
            config.get(CONF_PORT_MGMT_GRPC),
            config.get(CONF_PORT_MGMT_REST),
            config.get(CONF_PORT_INFER_GRPC),
            config.get(CONF_PORT_INFER_REST),
            config.get(CONF_TIMEOUT),
            models,
            targets,
            config.get(ATTR_CONFIDENCE),
            save_file_folder,
            config.get(CONF_SAVE_TIMESTAMPTED_FILE),
            config.get(CONF_SHOW_BOXES),
            config.get(CONF_SAVE_CROPPED_FILE),
            config.get(CONF_SAVE_LABEL_DATA),
            config.get(CONF_SAVE_BLANKS),
            config.get(CONF_SAVE_LATEST),
            camera.get(CONF_ENTITY_ID),
            config.get(CONF_FIRE_EVENTS),
            camera.get(CONF_NAME),
        )
        entities.append(object_entity)
        hass.data[DATA_TORCHSERVE].append(object_entity)
    add_devices(entities)

    response = eval(get_model_list_via_rest(config.get(CONF_IP_ADDRESS), config.get(CONF_PORT_MGMT_REST)))
    modellist = []
    for modelName, modelUrl in response["models"]:
        modellist.append(modelName)
    _LOGGER.info(f"Models detected on torchserve are {models}")


class ObjectClassifyEntity(ImageProcessingEntity):
    """Perform a face classification."""

    def __init__(
        self,
        ip_address,
        port_mgmt_grpc,
        port_mgmt_rest,
        port_infer_grpc,
        port_infer_rest,
        timeout,
        models,
        targets,
        confidence,
        save_file_folder,
        save_timestamped_file,
        show_boxes,
        save_cropped_file,
        save_label_data,
        save_blanks,
        save_latest,
        camera_entity,
        fire_events,
        name=None,
    ):
        """Init with the API key and model id."""
        super().__init__()

        self._ip_address = ip_address
        self._port_mgmt_grpc = port_mgmt_grpc
        self._port_mgmt_rest = port_mgmt_rest
        self._port_infer_grpc = port_infer_grpc
        self._port_infer_rest = port_infer_rest

        self._targets = targets
        self._models = models
        self._confidence = confidence
        self._camera = camera_entity
        if name:
            self._name = name
        else:
            camera_name = split_entity_id(camera_entity)[1]
            self._name = "torchserve_{}".format(camera_name)

        _LOGGER.debug("Torchserve camera initializing as {}".format(self._name))

        self._save_file_folder = save_file_folder
        self._save_timestamped_file = save_timestamped_file
        self._show_boxes = show_boxes
        self._save_cropped_file = save_cropped_file
        self._save_label_data = save_label_data
        self._save_blanks = save_blanks
        self._save_latest = save_latest
        self._fire_events = fire_events

        self._last_detection = None
        self._image_width = None
        self._image_height = None

        self._state = None
        self._objects = []  # The parsed raw data
        self._targets_found = []
        self._summary = {}
        self._matched = {}

    def get_inference_stub(self):
        """Get toirchserve GRPC stub."""
        channel = grpc.insecure_channel(f"{self._ip_address}:{self._port_infer_grpc}", options=[
            ('grpc.max_send_message_length', int(2147483647)),
            ('grpc.max_receive_message_length', int(2147483647)),
        ])
        stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
        return stub

    def get_management_stub(self):
        """Get toirchserve GRPC stub."""
        channel = grpc.insecure_channel(f"{self._ip_address}:{self._port_mgmt_grpc}")
        stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
        return stub

    def process_image(self, image):
        """Process an image."""
        img_byte_arr = io.BytesIO(bytearray(image))
        pil_image = Image.open(img_byte_arr)
        self._image_width, self._image_height = pil_image.size

        self._objects = []  # The parsed raw data
        self._targets_found = []
        self._summary = {}
        self._state = None

        for model in self._models:
            _LOGGER.debug(f"Torchserve model {model} detection triggering on {self._name}")
            tic = time.perf_counter()
            response = eval(infer_via_rest(self._ip_address, self._port_infer_rest, model, image))
            toc = time.perf_counter()

            if isinstance(response, str) or (isinstance(response, dict) and 'code' in response.keys()):
                # error?
                _LOGGER.critical(f"Torchsever unexpected response {response}")
                continue

            self._objects = get_objects(response, model, self._image_width, self._image_height)

            # filter confidence
            self._targets_found = [obj for obj in self._objects if (obj["confidence"] > self._confidence)]
            # filter classes
            if len(self._targets) > 0:
                self._targets_found = [obj for obj in self._targets_found if (obj["name"] in self._targets)]

            if len(self._targets_found) > 0:
                _LOGGER.info(f"Torchserve model {model} detection on {self._name} ran in {toc-tic}s and returned {response}")

            self._state = len(self._targets_found)

            detection_time = dt_util.now().strftime(DATETIME_FORMAT)
            if (self._state > 0):
                self._last_detection = detection_time

            if self._save_timestamped_file:
                self.save_image(self._save_file_folder, image, self._targets_found, detection_time)

            if (self._fire_events):
                objnum = 0
                for target in self._targets_found:
                    objnum = objnum + 1
                    target_event_data = target.copy()
                    target_event_data[ATTR_ENTITY_ID] = self.entity_id
                    target_event_data[ATTR_LAST_TRIP_TIME] = detection_time
                    target_event_data[CONF_COUNT] = objnum
                    target_event_data[DATA_MODEL] = model
                    target_event_type = "torchserve.{}_detected".format(target_event_data[DATA_PREDICTION_TYPE])
                    self.hass.bus.fire(target_event_type, target_event_data)
                    _LOGGER.debug(f"Torchserve event fired {target_event_data}")
                if objnum > 0:
                    _LOGGER.debug("Torchserve events fired")

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        return "targets"

    @property
    def should_poll(self):
        """Return the polling state."""
        return False

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        for target in self._targets:
            attr[f"{target} count"] = len(
                [t for t in self._objects if t["name"] == target]
            )
        if self._last_detection:
            attr["last_target_detection"] = self._last_detection
        attr["summary"] = self._summary
        attr["objects"] = self._objects
        return attr

    def save_image(self, directory, image, objects, stamp):
        """Save image files."""
        _LOGGER.debug("Torchserve saving files")
        """Draws the actual bounding box of the detected objects."""
        img = Image.open(io.BytesIO(bytearray(image))).convert("RGB")
        imgc = img.copy()

        if self._show_boxes:
            draw = ImageDraw.Draw(img)

        prefix = f"{get_valid_filename(self._name).lower()}"

        objnum = 1
        for obj in objects:
            name = obj["name"]
            confidence = obj["confidence"]
            model = obj["model"]
            box_label = f"{name}: {confidence:.1f}%"
            prediction_type = obj["prediction_type"]

            if "bounding_box" in obj:
                box = obj["bounding_box"]
                centroid = obj["centroid"]

                if self._save_cropped_file:
                    imc = imgc.crop((box["x_min"] * img.width, box["y_min"] * img.height, box["x_max"] * img.width, box["y_max"] * img.height))
                    if self._save_timestamped_file:
                        crop_save_path = directory / f"{prefix}_{stamp}_{model}_{prediction_type}_{name}_{objnum}.jpg"
                        imc.save(crop_save_path)
                    if self._save_latest:
                        crop_save_path = directory / f"{prefix}_latest_{model}_{prediction_type}_{name}.jpg"
                        imc.save(crop_save_path)
                    _LOGGER.debug("Torchserve saved crops")

            if self._save_label_data:
                label_path = directory / "labels.csv"
                with open(label_path, "a+") as f:
                    timestamp_save_path = directory / f"{self._name}_{stamp}_nobox.jpg"
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(timestamp_save_path, int(box["x_min"] * img.width), int(box["y_min"] * img.height), int(box["x_max"] * img.width), int(box["y_max"] * img.height), model, name, objnum, confidence))
                _LOGGER.debug("Torchserve saved labels")

            objnum = objnum + 1

            if self._show_boxes:
                box_colour = YELLOW

                draw_box(
                    draw,
                    (box["y_min"], box["x_min"], box["y_max"], box["x_max"]),
                    img.width,
                    img.height,
                    text=box_label,
                    color=box_colour,
                )

                # draw bullseye
                draw.text(
                    (centroid["x"] * img.width, centroid["y"] * img.height),
                    text="X",
                    fill=box_colour,
                )

        if (len(objects) > 0 or self._save_blanks):
            suffix = ""
            if (self._save_blanks and not len(objects) > 0):
                suffix = "_blank"

            if self._save_latest:
                if self._show_boxes:
                    latest_save_path = (directory / f"{prefix}_latest_box{suffix}.jpg")
                    img.save(latest_save_path)
                latest_save_path = directory / f"{prefix}_latest_nobox{suffix}.jpg"
                imgc.save(latest_save_path)

            if self._save_timestamped_file:
                if self._show_boxes:
                    timestamp_save_path = directory / f"{prefix}_{stamp}_box{suffix}.jpg"
                    img.save(timestamp_save_path)
                timestamp_save_path = directory / f"{prefix}_{stamp}_nobox{suffix}.jpg"
                imgc.save(timestamp_save_path)
            _LOGGER.debug("Torchserve saved uncropped images")
