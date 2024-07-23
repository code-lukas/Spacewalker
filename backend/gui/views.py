import numpy as np
import pandas as pd
import os
import joblib
import tritonclient.utils
from django.core import serializers
from django.contrib import messages
from django.shortcuts import render, redirect, reverse
from django.views.generic import TemplateView
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect, JsonResponse
from django.utils.decorators import method_decorator
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import requires_csrf_token, csrf_exempt
from functools import partial
from hnne import HNNE
import ujson as json

from tempfile import NamedTemporaryFile
from threading import Thread
from queue import Queue
from typing import Any

import cv2 as cv
# from minio.select import CSVInputSerialization, CSVOutputSerialization, SelectRequest
from .models import DataPoint, AnnotationColorDescription
import multiprocessing as mp
from sklearn.manifold import MDS, Isomap
from umap import UMAP
from openTSNE import TSNE
from sklearn.decomposition import PCA
from ..services.triton_inference import triton_inference, triton_inference_text
from ..services.minio_interface import MinioClient
from .forms import ConfigurationForm, InferenceSettingsForm

import logging

MAX_PROCESSES = 10
SAMPLE_N_FRAMES_VIDEO = 10
IMAGENET_MEANS = np.array([0.485, 0.456, 0.406])
IMAGENET_STDS = np.array([0.229, 0.224, 0.225])

size_lut = {
    'vit': 1024,
    'resnet50': 224,
    'vgg16': 224,
    'dinov2': 224,
    'clip_image': 224,
    'clip_video': 224,
}
triton_name_lut = {
    'vit': 'vit_b_onnx',
    'resnet50': 'resnet_50_onnx',
    'vgg16': 'vgg_16_onnx',
    'dinov2': 'dinov2_vitb14_onnx',
    'minilm_l6_v2': 'MiniLM_ensemble',
    'clip_image': 'CLIP_image',
    'clip_text': 'CLIP_text',
    'clip_video': 'CLIP_video',
}


def resize_longest_edge(img: np.array, longest_edge: int) -> np.array:
    h, w, _ = img.shape
    if h > w:
        new_w = int((longest_edge / h) * w)
        new_h = longest_edge
    else:
        new_h = int((longest_edge / w) * h)
        new_w = longest_edge
    return cv.resize(img, (new_h, new_w), interpolation=cv.INTER_CUBIC)


def prepare_image_for_inference(image: np.array, model: str):
    # I should consider building this into a proper dataloader
    if image.ndim != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    h = image.shape[0]
    w = image.shape[1]
    width = height = size_lut[model.lower()]

    if h > height or w > width:
        image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC).astype(np.uint8)
    else:
        a = (width - h) // 2
        aa = width - a - h

        b = (height - w) // 2
        bb = height - b - w

        image = np.pad(image, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant').astype(np.uint8)

    # scaling
    image -= image.min(axis=(0, 1), keepdims=True)
    image = image.astype(np.float32)
    image /= image.max(axis=(0, 1), keepdims=True)

    # imagenet normalization
    image = (image - IMAGENET_MEANS) / IMAGENET_STDS

    return np.moveaxis(image, [0, 1, 2], [1, 2, 0])


# Create your views here.

class Connector:
    def __init__(self,
                 minio_url: str = 'minio:9000',
                 minio_credentials: tuple[str, str] = ('demo', 'demodemo')
                 ) -> None:
        self.minio_client = MinioClient(
            endpoint=minio_url,
            access_key=minio_credentials[0],
            secret_key=minio_credentials[1]
        )


class guiView(Connector, TemplateView):
    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        context = {}

        project, model, dr_method = request.GET.get('project').split(' ')
        context['project'] = project
        data3d = json.loads(serializers.serialize('json', DataPoint.objects.filter(
            project_name=context['project'],
            model=model,
            dr_method=dr_method,
            is3d=True).order_by('file_reference')))
        data3d = [d['fields'] for d in data3d]
        context['data3d'] = mark_safe(json.dumps(data3d, escape_forward_slashes=False))

        data2d = json.loads(serializers.serialize('json', DataPoint.objects.filter(
            project_name=context['project'],
            model=model,
            dr_method=dr_method,
            is3d=False).order_by('file_reference')))
        data2d = [d['fields'] for d in data2d]
        context['data2d'] = mark_safe(json.dumps(data2d, escape_forward_slashes=False))

        label_lut = json.loads(serializers.serialize('json', AnnotationColorDescription.objects.filter(
            project_association=project,
        )))
        context['existing_labels'] = mark_safe(json.dumps(label_lut, escape_forward_slashes=False))

        template_name = "gui/gui.html"
        return render(request, template_name, context=context)

    @method_decorator(requires_csrf_token)
    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:

        if 'requestType' in request.POST:
            # can't use both image and text
            if (len(request.FILES) != 0) and ('textInput' in request.POST):
                return HttpResponse(500)

            used_model = request.POST['model']
            project_name = request.POST['project']

            # FIXME: The model matching for text -> image and image -> text is hardcoded, if we get more models i should
            # build a LUT
            # this means an image query
            if len(request.FILES) != 0:
                image = cv.imread(request.FILES['imageInput'].temporary_file_path())
                # This likely means an image query in a text based project
                if used_model not in size_lut:
                    used_model = 'CLIP_image'
                image = prepare_image_for_inference(image, model=used_model)

                embedding = triton_inference(image.astype(np.float32), model_name=triton_name_lut[used_model.lower()])
            else:
                # text query
                text = request.POST['textInput']
                embedding = triton_inference_text(text, model_name="CLIP_text")

            # load the dr method
            with NamedTemporaryFile(mode='wb', suffix='.pkl') as dr_method:
                self.minio_client.client.fget_object(
                    bucket_name='spacewalker-projects',
                    object_name=f'{project_name}/dr/dr2d.pkl',
                    file_path=dr_method.name
                )
                with open(dr_method.name, 'rb') as f:
                    save_data = joblib.load(f)
                query2d = save_data['dr_method'].transform(embedding)
                query2d /= float(save_data['scale_val'])

                with NamedTemporaryFile(mode='wb', suffix='.pkl') as dr_method:
                    self.minio_client.client.fget_object(
                        bucket_name='spacewalker-projects',
                        object_name=f'{project_name}/dr/dr3d.pkl',
                        file_path=dr_method.name
                    )
                    with open(dr_method.name, 'rb') as f:
                        save_data = joblib.load(f)
                    query3d = save_data['dr_method'].transform(embedding)
                    query3d /= float(save_data['scale_val'])

            embedding = {
                '2d_embedding': tuple(query2d[0].astype(np.float64)),
                '3d_embedding': tuple(query3d[0].astype(np.float64)),
            }
            return JsonResponse(embedding, status=200)
        else:
            three_js_data = json.loads(request.body)
            with mp.Pool(MAX_PROCESSES) as pool:
                pool.map(self.update_datapoint, iterable=three_js_data)
            return HttpResponse(200)

    @staticmethod
    def update_datapoint(datapoint: dict):
        # Normally we'd have to differentiate between 2D and 3D points, but in this case we don't care as they get the
        # same label
        obj_1, obj_2 = DataPoint.objects.filter(file_reference=datapoint['file_reference'])
        obj_1.cluster_id = datapoint['cluster_id']
        obj_2.cluster_id = datapoint['cluster_id']
        obj_1.save()
        obj_2.save()


class configurationView(Connector, TemplateView):
    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        template_name = 'gui/index.html'
        context = {
            'form': ConfigurationForm()
        }
        return render(request, template_name, context=context)

    @method_decorator(requires_csrf_token)
    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        form: ConfigurationForm = ConfigurationForm(request.POST, request.FILES)
        if 'input_path' in request.FILES:
            form.fields['input_path'].choices = [(request.FILES['input_path'], request.FILES['input_path'])]
            files = request.FILES.getlist('input_path')
            files = [{'fp': f.temporary_file_path(), 'name': f.name} for f in files]
        else:
            files = []

        if form.is_valid():
            project_name = request.POST['project_name']
            for file in files:
                self.minio_client.send_to_bucket(
                    bucket_name='spacewalker-projects',
                    file=file['fp'],
                    directory=f'{project_name}/raw',
                    name_on_storage=file['name']
                )
            messages.success(request, 'Submission successful!')
        else:
            messages.error(request, 'Submission unsuccessful! - Please check the file types')
        return HttpResponseRedirect(request.path_info)


class InferenceSettingsView(Connector, TemplateView):
    def get_data(self, file):
        with NamedTemporaryFile(mode='wb', suffix='.png') as temporary_image:
            self.minio_client.client.fget_object(
                bucket_name='spacewalker-projects',
                object_name=file,
                file_path=temporary_image.name
            )
            return np.load(temporary_image.name)

    @staticmethod
    def make_point(
            model: str,
            dr_method: str,
            project_name: str,
            modality: str,
            point: np.array,
            filename: str,
            thumbnail_name: str,
    ) -> DataPoint:
        is3d = len(point) == 3
        z = 0
        if is3d:
            z = point[2]
        return DataPoint(
            x=point[0],
            y=point[1],
            z=z,
            model=model,
            dr_method=dr_method,
            project_name=project_name,
            class_name='',
            cluster_id=0,
            file_reference=filename,
            preview=thumbnail_name,
            is3d=is3d,
            modality=modality,
        )

    def inference(self, project: str, model: str, dr_method: str) -> None:
        file_gen = self.minio_client.client.list_objects(
            bucket_name='spacewalker-projects',
            prefix=f'{project}/raw',
            recursive=True
        )
        files = [f.object_name for f in file_gen]

        # Distinguish between text, image and video
        file_type = files[0].split('.')[-1].lower()

        modality = None
        match file_type:
            case 'png' | 'jpg' | 'jpeg':
                modality = 'image'
                for file in files:
                    with NamedTemporaryFile(mode='wb', suffix='.png') as temporary_image:
                        self.minio_client.client.fget_object(
                            bucket_name='spacewalker-projects',
                            object_name=file,
                            file_path=temporary_image.name
                        )
                        image = cv.imread(temporary_image.name)
                        image = prepare_image_for_inference(image, model.lower())
                        # images need to be padded to fit vit input shape
                        result = triton_inference(
                            image_data=image.astype(np.float32),
                            model_name=triton_name_lut[model.lower()]
                        )
                        with NamedTemporaryFile(mode='w', suffix='.npy') as npy_temp:
                            np.save(npy_temp.name, result)
                            new_fn = file.split('/')[-1].split('.')[0] + '.upload.npy'
                            self.minio_client.send_to_bucket(
                                bucket_name='spacewalker-projects',
                                file=npy_temp.name,
                                directory=f'{project}/npy',
                                name_on_storage=new_fn
                            )
            case 'csv':
                modality = 'text'
                master_file = files[0]
                # FIXME: This is a way more elegant solution, however the parsing needs to be more robust
                # with self.minio_client.client.select_object_content(
                #         bucket_name='spacewalker-projects',
                #         object_name=master_file,
                #         request=SelectRequest(
                #             "select * from S3Object",
                #             CSVInputSerialization(),
                #             CSVOutputSerialization(),
                #             request_progress=True)) as result:
                #     for data in result.stream():
                #         lines = data.split(b'\n')
                #         for line in lines:
                #             text_id, text = line.decode().split(',', 1)
                #             text_id = int(text_id)

                with NamedTemporaryFile(mode='wb', suffix='.csv') as temporary_text:
                    self.minio_client.client.fget_object(
                        bucket_name='spacewalker-projects',
                        object_name=master_file,
                        file_path=temporary_text.name
                    )
                    df = pd.read_csv(temporary_text.name)
                    for row in df.itertuples():
                        try:
                            result = triton_inference_text(
                                text=row.Text,
                                model_name=triton_name_lut[model.lower()]
                            )
                        except tritonclient.utils.InferenceServerException:
                            logging.error(f'Error encountered in {row.Text}')

                        with NamedTemporaryFile(mode='w', suffix='.npy') as npy_temp:
                            np.save(npy_temp.name, result)
                            new_fn = f'{row.Id}.upload.npy'
                            self.minio_client.send_to_bucket(
                                bucket_name='spacewalker-projects',
                                file=npy_temp.name,
                                directory=f'{project}/npy',
                                name_on_storage=new_fn
                            )
            case 'mp3' | 'mp4':
                modality = 'video'
                for file in files:
                    framestack = []
                    with NamedTemporaryFile(mode='wb', suffix='.mp4') as temporary_video:
                        self.minio_client.client.fget_object(
                            bucket_name='spacewalker-projects',
                            object_name=file,
                            file_path=temporary_video.name
                        )
                        # get frames to build batch
                        video = cv.VideoCapture(temporary_video.name)
                        frame_count = video.get(cv.CAP_PROP_FRAME_COUNT)
                        equidistant_frame_indices = np.arange(0, frame_count, frame_count/SAMPLE_N_FRAMES_VIDEO)\
                            .astype(int)
                        for frame in equidistant_frame_indices:
                            video.set(cv.CAP_PROP_POS_FRAMES, frame)
                            ret, image = video.read()
                            image = prepare_image_for_inference(image, model.lower())
                            framestack.append(image)
                        framestack = np.array(framestack).astype(np.float32)
                        result = triton_inference(
                            image_data=framestack.astype(np.float32),
                            model_name=triton_name_lut[model.lower()]
                        )

                    with NamedTemporaryFile(mode='w', suffix='.npy') as npy_temp:
                        np.save(npy_temp.name, result)
                        new_fn = file.split('/')[-1].split('.')[0] + '.upload.npy'
                        self.minio_client.send_to_bucket(
                            bucket_name='spacewalker-projects',
                            file=npy_temp.name,
                            directory=f'{project}/npy',
                            name_on_storage=new_fn
                        )
        self.start_dr(project=project, model=model.lower(), dr_method=dr_method, modality=modality)

    def start_dr(self, project: str, model: str, dr_method: str, modality: str) -> None:
        project_files = self.minio_client.client.list_objects(
            bucket_name='spacewalker-projects',
            prefix=f'{project}/npy',
            recursive=True
        )
        fns = [i.object_name for i in list(project_files)]
        # TODO: Multiprocess this !
        data = list(map(self.get_data, fns))

        match modality.lower():
            case 'image':
                fns = [i.replace('/npy/', '/raw/').replace('.npy', '.png') for i in fns]
                thumbs = [i.replace('/raw/', '/thumbs/').replace('upload', 'thumb') for i in fns]
            case 'text':
                fns = [i.replace('/npy/', '/thumbs/').replace('.npy', '.txt').replace('.upload', '') for i in fns]
                thumbs = fns
            case 'video':
                fns = [i.replace('/npy/', '/raw/').replace('.npy', '.png') for i in fns]
                thumbs = [i.replace('/raw/', '/thumbs/').replace('upload', 'thumb') for i in fns]
            case _:
                raise ValueError(f'Unknown {modality=}')

        x = np.array(data).squeeze()
        match dr_method.lower():
            case 'hnne':
                dr2d = HNNE(dim=2)
                dr3d = HNNE(dim=3)
            case 'umap':
                dr2d = UMAP(n_components=2, n_jobs=-1)
                dr3d = UMAP(n_components=3, n_jobs=-1)
            case 'pca':
                dr2d = PCA(n_components=2)
                dr3d = PCA(n_components=3)
            case 't-sne':
                dr2d = TSNE(n_components=2, n_jobs=-1)
                dr3d = TSNE(n_components=3, n_jobs=-1, negative_gradient_method='bh')
            case 'mds':
                dr2d = MDS(n_components=2, normalized_stress='auto', n_jobs=-1)
                dr3d = MDS(n_components=3, normalized_stress='auto', n_jobs=-1)
            case 'isomap':
                dr2d = Isomap(n_components=2, n_jobs=-1)
                dr3d = Isomap(n_components=2, n_jobs=-1)
            case _:
                raise RuntimeError(f'Dimensionality reduction method {dr_method} unknown')
        scale = 1
        # NOTE: This line will fail if there are too few datapoints!
        if dr_method.lower() == 'hnne':
            proj2d = dr2d.fit_transform(x, dim=2)
        else:
            dr2d = dr2d.fit(x)
            proj2d = dr2d.transform(x)

        scale_val_2d = np.max(proj2d)

        # dr2d = (dr2d / np.linalg.norm(dr2d)) * scale
        with NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as dr2d_save:
            save_data = {
                'dr_method': dr2d,
                'scale_val': scale_val_2d
            }
            joblib.dump(save_data, dr2d_save)

        self.minio_client.send_to_bucket(
            bucket_name='spacewalker-projects',
            file=dr2d_save.name,
            directory=f'{project}/dr',
            name_on_storage='dr2d.pkl'
        )
        os.remove(dr2d_save.name)

        if dr_method.lower() == 'hnne':
            proj3d = dr3d.fit_transform(x, dim=3)
        else:
            dr3d = dr3d.fit(x)
            proj3d = dr3d.transform(x)

        scale_val_3d = np.max(proj3d)

        # dr3d = (dr3d / np.linalg.norm(dr3d)) * scale
        with NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as dr3d_save:
            save_data = {
                'dr_method': dr3d,
                'scale_val': scale_val_3d
            }
            joblib.dump(save_data, dr3d_save)

        self.minio_client.send_to_bucket(
            bucket_name='spacewalker-projects',
            file=dr3d_save.name,
            directory=f'{project}/dr',
            name_on_storage='dr3d.pkl'
        )
        os.remove(dr3d_save.name)

        proj2d = (proj2d / scale_val_2d) * scale
        proj3d = (proj3d / scale_val_3d) * scale

        points = list(proj2d) + list(proj3d)
        point_creation = partial(self.make_point, model, dr_method, project, modality)
        with mp.Pool(processes=MAX_PROCESSES) as pool:
            new_points = pool.starmap(point_creation, zip(points, fns * 2, thumbs * 2))
        DataPoint.objects.bulk_create(new_points)

    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        template_name = 'gui/inference_settings.html'
        dynamic_choices = [(name := i.object_name.rstrip('/'), name) for i in self.minio_client.client.list_objects(
            bucket_name='spacewalker-projects',
            recursive=False
        )]
        context = {
            'form': InferenceSettingsForm(dynamic_choices=dynamic_choices)
        }
        return render(request, template_name, context=context)

    @method_decorator(requires_csrf_token)
    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        messages.success(request, 'Inference starting!')
        model = request.POST['model']
        dr_method = request.POST['dr_method']
        project = request.POST['project']
        task = Thread(target=self.inference, args=(project, model, dr_method,))
        task.start()
        return HttpResponseRedirect(request.path_info)


class MinIOWebhook(Connector, TemplateView):
    def __init__(self, **kwargs: Any) -> None:
        self.thumbnail_maxsize = 128
        self.thumbnail_tasks = Queue()
        self.consumer = Thread(target=self.thumbnail_handler, args=(self.thumbnail_tasks,))
        super().__init__(**kwargs)

    def thumbnail_handler(self, fn: str):
        # Just check if server is ready: curl -v localhost:8000/v2/health/ready
        # else wait, then go again
        file = '/'.join([part for part in fn.split('/')[1:]])
        project_name = file.split('/')[0]
        with NamedTemporaryFile(mode='wb', suffix='.png') as temporary_image:
            self.minio_client.client.fget_object(
                bucket_name='spacewalker-projects',
                object_name=file,
                file_path=temporary_image.name
            )
            # I should consider building this into a proper dataloader
            image = cv.imread(temporary_image.name)
            if image.ndim == 3:
                # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                pass
            else:
                image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

            # make thumbnail
            thumbnail = resize_longest_edge(image, self.thumbnail_maxsize).astype(np.uint8)

            with NamedTemporaryFile(mode='w', suffix='.png') as thumb:
                cv.imwrite(thumb.name, thumbnail)
                thumbnail_name = file.split('/')[-1].split('.')[0] + '.thumb.png'
                self.minio_client.send_to_bucket(
                    bucket_name='spacewalker-projects',
                    file=thumb.name,
                    directory=f'{project_name}/thumbs',
                    name_on_storage=thumbnail_name
                )

    def text_preview_handler(self, fn: str):
        file = '/'.join([part for part in fn.split('/')[1:]])
        project_name = file.split('/')[0]
        with NamedTemporaryFile(mode='wb', suffix='.csv') as temporary_table:
            self.minio_client.client.fget_object(
                bucket_name='spacewalker-projects',
                object_name=file,
                file_path=temporary_table.name
            )
            df = pd.read_csv(temporary_table.name)
            for row in df.itertuples():
                with NamedTemporaryFile(mode='w+', suffix='.txt') as txt:
                    with open(txt.name, 'w') as txt_file:
                        txt_file.write(row.Text)
                    self.minio_client.send_to_bucket(
                        bucket_name='spacewalker-projects',
                        file=txt.name,
                        directory=f'{project_name}/thumbs',
                        name_on_storage=f'{row.Id}.txt'
                    )

    def video_preview_handler(self, fn: str):
        file = '/'.join([part for part in fn.split('/')[1:]])
        project_name = file.split('/')[0]
        with NamedTemporaryFile(mode='wb', suffix='.mp4') as temporary_video:
            self.minio_client.client.fget_object(
                bucket_name='spacewalker-projects',
                object_name=file,
                file_path=temporary_video.name
            )
            video = cv.VideoCapture(temporary_video.name)
            video.set(cv.CAP_PROP_POS_FRAMES, 0)
            _, frame = video.read()
            thumbnail = resize_longest_edge(frame, self.thumbnail_maxsize).astype(np.uint8)

            with NamedTemporaryFile(mode='w', suffix='.png') as thumb:
                cv.imwrite(thumb.name, thumbnail)
                thumbnail_name = file.split('/')[-1].split('.')[0] + '.thumb.png'
                self.minio_client.send_to_bucket(
                    bucket_name='spacewalker-projects',
                    file=thumb.name,
                    directory=f'{project_name}/thumbs',
                    name_on_storage=thumbnail_name
                )

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs) -> None:
        return super(MinIOWebhook, self).dispatch(*args, **kwargs)

    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        return HttpResponse(404)

    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        message = json.loads(request.body)
        match message['EventName']:
            case 's3:ObjectCreated:Put':
                if (message['Key'].endswith('.png')) and ('thumb' not in message['Key'].lower()):
                    task = Thread(target=self.thumbnail_handler, args=(message['Key'],))
                    task.daemon = True
                    task.start()
                elif message['Key'].endswith('.csv'):
                    task = Thread(target=self.text_preview_handler, args=(message['Key'],))
                    task.daemon = True
                    task.start()
            case 's3:ObjectCreated:CompleteMultipartUpload':
                if message['Key'].endswith('.mp4'):
                    task = Thread(target=self.video_preview_handler, args=(message['Key'],))
                    task.daemon = True
                    task.start()
                elif message['Key'].endswith('.csv'):
                    task = Thread(target=self.text_preview_handler, args=(message['Key'],))
                    task.daemon = True
                    task.start()
            case _:
                pass
        return HttpResponse(200)


class ProjectSelectionView(TemplateView):
    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        projects = list(DataPoint.objects.values_list('project_name', flat=True).distinct())
        combinations = {}
        for project in projects:
            combinations[project] = {
                'combination': [],
            }
            settings = list(DataPoint.objects.values_list('model', 'dr_method').filter(project_name=project).distinct())
            for setting in settings:
                combinations[project]['combination'].append(f'{setting[0]}+{setting[1]}')

        all_labels = json.loads(serializers.serialize('json', AnnotationColorDescription.objects.all()))
        all_labels = [i['fields'] for i in all_labels]

        context = {
            'dropdown_items': projects,
            'combinations': mark_safe(json.dumps(combinations, escape_forward_slashes=False)),
            'all_labels': mark_safe(json.dumps(all_labels, escape_forward_slashes=False))
        }
        return render(request, 'gui/project_selection.html', context)

    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        selected_option = request.POST.get('dropdown', None)
        setting = request.POST.get('setting', None)
        labels = json.loads(request.POST.get('labels', None))

        model, dr_method = setting.split('+')

        labels_in_db = AnnotationColorDescription.objects.filter(project_association=selected_option)
        for existing_label in labels_in_db:
            existing_label.delete()

        for label in labels:
            AnnotationColorDescription.objects.update_or_create(
                project_association=selected_option,
                class_id=int(label['id']),
                name=label['name'],
                color=label['color']
            )

        redirect_url = reverse('gui')
        redirect_url += f'?project={selected_option}+{model}+{dr_method}'  # Append the selected option as a query parameter
        return redirect(redirect_url)


class ExportView(TemplateView):
    def get(self, request, *args, **kwargs):
        project = request.GET.get('project')
        data = serializers.serialize('json', DataPoint.objects.filter(project_name=project))

        response = HttpResponse(data, content_type='application/json')
        response['Content-Disposition'] = 'attachment; filename="exported_data.json"'
        return response
