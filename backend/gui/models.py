from django.db import models


# Create your models here.
class DataPoint(models.Model):
    # django automatically adds a primary key
    # id = models.BigAutoField(primary_key=True)
    x = models.FloatField()
    y = models.FloatField()
    z = models.FloatField()
    is3d = models.BooleanField()
    model = models.CharField()
    dr_method = models.CharField()
    project_name = models.CharField()
    class_name = models.CharField(max_length=50)
    cluster_id = models.BigIntegerField()
    file_reference = models.CharField()
    preview = models.CharField()
    modality = models.CharField()

    @classmethod
    def create(cls,
               x: float,
               y: float,
               z: float | None,
               model: str,
               dr_method: str,
               project_name: str,
               class_name: str | None,
               cluster_id: int,
               file_reference: str,
               preview: str,
               is3d: bool,
               modality: str,
               ):
        datapoint = cls(
            x=x,
            y=y,
            z=z,
            model=model,
            dr_method=dr_method,
            project_name=project_name,
            class_name=class_name,
            cluster_id=cluster_id,
            file_reference=file_reference,
            preview=preview,
            is3d=is3d,
            modality=modality,
        )
        return datapoint


class AnnotationColorDescription(models.Model):
    project_association = models.CharField()
    class_id = models.IntegerField()
    color = models.CharField(max_length=7)  # #ffffff
    name = models.CharField()

    @classmethod
    def create(cls,
               project_association: str,
               class_id: int,
               color: str,
               name: str,
               ):
        return cls(
            project_association=project_association,
            class_id=class_id,
            color=color,
            name=name
        )
