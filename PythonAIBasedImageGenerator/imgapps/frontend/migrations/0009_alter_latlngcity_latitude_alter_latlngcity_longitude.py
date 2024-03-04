# Generated by Django 4.2.3 on 2023-08-04 06:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frontend', '0008_rename_longitute_latlngcity_longitude'),
    ]

    operations = [
        migrations.AlterField(
            model_name='latlngcity',
            name='Latitude',
            field=models.DecimalField(decimal_places=6, max_digits=10),
        ),
        migrations.AlterField(
            model_name='latlngcity',
            name='Longitude',
            field=models.DecimalField(decimal_places=6, max_digits=10),
        ),
    ]
