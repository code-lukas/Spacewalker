from django import forms
from ..services.minio_interface import MinioClient

MODEL_CHOICES = [
    ('ResNet50', 'ResNet-50'),
    ('VGG16', 'VGG-16'),
    ('ViT', 'ViT'),
    ('DINOV2', 'DINOV2'),
]
DR_METHODS = [
    ('HNNE', 'HNNE'),
    ('PCA', 'PCA'),
    ('T-SNE', 'T-SNE'),
    ('MDS', 'MDS'),
    ('ISOMAP', 'ISOMAP'),
]


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


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class ConfigurationForm(forms.Form):
    project_name = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label='Project name',
        required=True
    )

    input_path = MultipleFileField(
        required=False,
        label='Add / upload files to project',
    )


class InferenceSettingsForm(forms.Form):
    def __init__(self, *args, **kwargs):
        dynamic_choices = kwargs.pop('dynamic_choices', None)
        super(InferenceSettingsForm, self).__init__(*args, **kwargs)

        # Add a choice field dynamically if dynamic_choices is provided
        if dynamic_choices:
            self.fields['project'] = forms.ChoiceField(
                choices=dynamic_choices,
                widget=forms.Select(attrs={'class': 'form-control'}),
            )

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Model'
    )

    dr_method = forms.ChoiceField(
        choices=DR_METHODS,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Dimensionality reduction method'
    )
