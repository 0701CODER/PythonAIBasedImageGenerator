# Generated by Django 4.2.3 on 2023-08-04 07:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frontend', '0009_alter_latlngcity_latitude_alter_latlngcity_longitude'),
    ]

    operations = [
        migrations.AddField(
            model_name='latlngcity',
            name='Zone',
            field=models.CharField(choices=[('Green', 'Green'), ('Red', 'Red'), ('Blue', 'Blue')], default='Green', max_length=200),
        ),
    ]
