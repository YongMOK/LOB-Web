from django import template
from django.urls import resolve

register = template.Library()

@register.simple_tag
def active(request, url):
    if resolve(request.path_info).url_name == url:
        return 'active'
    return ''