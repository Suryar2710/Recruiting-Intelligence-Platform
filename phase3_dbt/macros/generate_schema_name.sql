{#
    Override dbt's default schema naming.

    Default behavior concatenates the profile schema with the model's custom
    schema (e.g. "public" + "staging" => "public_staging"). For a clean,
    readable warehouse we want the custom schema used verbatim: staging,
    intermediate, marts. When a model declares no custom schema, fall back to
    the profile's target schema.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
