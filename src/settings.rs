// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! DataFusion-native session settings forwarded to VGI workers.
//!
//! DataFusion requires third-party settings to live under a configuration
//! namespace. `VgiSettings` therefore exposes `vgi.<worker_setting>` while the
//! wire batch retains the worker's original unprefixed field names.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
use datafusion::common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion::common::{DataFusionError, Result as DFResult, ScalarValue};

/// Dynamic VGI worker settings stored in DataFusion's session configuration.
///
/// Values use DataFusion's ordinary `SET` string representation and are cast
/// to each attached worker's advertised Arrow type immediately before bind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VgiSettings {
    values: BTreeMap<String, String>,
}

impl VgiSettings {
    /// Set one worker setting without going through SQL.
    pub fn set_value(&mut self, name: impl Into<String>, value: impl Into<String>) -> DFResult<()> {
        let name = normalize_name(&name.into())?;
        self.values.insert(name, value.into());
        Ok(())
    }

    /// Remove one worker setting, restoring worker/default behavior.
    pub fn reset_value(&mut self, name: &str) -> bool {
        self.values.remove(&name.to_ascii_lowercase()).is_some()
    }

    /// Inspect the configured string value.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.values
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub(crate) fn values(&self) -> &BTreeMap<String, String> {
        &self.values
    }
}

impl ExtensionOptions for VgiSettings {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> DFResult<()> {
        self.set_value(key, value)
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        self.values
            .iter()
            .map(|(name, value)| ConfigEntry {
                key: format!("vgi.{name}"),
                value: Some(value.clone()),
                description: "Value forwarded using the attached VGI worker's declared Arrow type",
            })
            .collect()
    }
}

impl ConfigExtension for VgiSettings {
    const PREFIX: &'static str = "vgi";
}

fn normalize_name(name: &str) -> DFResult<String> {
    let name = name.trim();
    if name.is_empty() || name.contains('.') {
        return Err(DataFusionError::Configuration(format!(
            "VGI setting names must be non-empty and contain no dots, found `{name}`"
        )));
    }
    Ok(name.to_ascii_lowercase())
}

pub(crate) fn encode_settings(
    configured: &VgiSettings,
    declarations: &[vgi_client::SettingSpec],
) -> DFResult<Option<vgi_client::Bytes>> {
    let mut fields = Vec::new();
    let mut arrays = Vec::<ArrayRef>::new();
    for declaration in declarations {
        let Some(raw) = configured.get(&declaration.name) else {
            continue;
        };
        let scalar = parse_setting_value(raw, &declaration.data_type).map_err(|error| {
            DataFusionError::Configuration(format!(
                "VGI setting `{}` cannot be cast to {}: {error}",
                declaration.name, declaration.data_type
            ))
        })?;
        fields.push(Field::new(
            &declaration.name,
            declaration.data_type.clone(),
            scalar.is_null(),
        ));
        arrays.push(scalar.to_array_of_size(1)?);
    }
    if fields.is_empty() {
        return Ok(None);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    vgi_protocol::ipc::write_batch(&batch)
        .map(vgi_client::Bytes::from)
        .map(Some)
        .map_err(crate::to_df)
}

fn parse_setting_value(raw: &str, data_type: &DataType) -> DFResult<ScalarValue> {
    match data_type {
        DataType::Utf8 => Ok(ScalarValue::Utf8(Some(raw.to_string()))),
        DataType::LargeUtf8 => Ok(ScalarValue::LargeUtf8(Some(raw.to_string()))),
        DataType::Utf8View => Ok(ScalarValue::Utf8View(Some(raw.to_string()))),
        DataType::Struct(fields) => parse_struct(raw, fields),
        _ => ScalarValue::try_from_string(raw.to_string(), data_type),
    }
}

fn parse_struct(raw: &str, fields: &Fields) -> DFResult<ScalarValue> {
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|error| {
        DataFusionError::Configuration(format!("struct settings use a JSON object string: {error}"))
    })?;
    let object = value.as_object().ok_or_else(|| {
        DataFusionError::Configuration("struct settings use a JSON object string".to_string())
    })?;
    for name in object.keys() {
        if !fields.iter().any(|field| field.name() == name) {
            return Err(DataFusionError::Configuration(format!(
                "unknown struct setting field `{name}`"
            )));
        }
    }
    let arrays = fields
        .iter()
        .map(|field| {
            let scalar = match object.get(field.name()) {
                None | Some(serde_json::Value::Null) if field.is_nullable() => {
                    ScalarValue::try_new_null(field.data_type())?
                }
                None | Some(serde_json::Value::Null) => {
                    return Err(DataFusionError::Configuration(format!(
                        "struct setting field `{}` is required",
                        field.name()
                    )))
                }
                Some(value) => parse_json_scalar(value, field.data_type())?,
            };
            scalar.to_array_of_size(1)
        })
        .collect::<DFResult<Vec<_>>>()?;
    Ok(ScalarValue::Struct(Arc::new(StructArray::new(
        fields.clone(),
        arrays,
        None,
    ))))
}

fn parse_json_scalar(value: &serde_json::Value, data_type: &DataType) -> DFResult<ScalarValue> {
    match (value, data_type) {
        (serde_json::Value::String(value), DataType::Utf8) => {
            Ok(ScalarValue::Utf8(Some(value.clone())))
        }
        (serde_json::Value::String(value), DataType::LargeUtf8) => {
            Ok(ScalarValue::LargeUtf8(Some(value.clone())))
        }
        (serde_json::Value::String(value), DataType::Utf8View) => {
            Ok(ScalarValue::Utf8View(Some(value.clone())))
        }
        (serde_json::Value::Object(_), DataType::Struct(fields)) => {
            parse_struct(&value.to_string(), fields)
        }
        (serde_json::Value::String(value), _) => {
            ScalarValue::try_from_string(value.clone(), data_type)
        }
        (value @ (serde_json::Value::Bool(_) | serde_json::Value::Number(_)), _) => {
            ScalarValue::try_from_string(value.to_string(), data_type)
        }
        _ => Err(DataFusionError::Configuration(format!(
            "JSON value {value} is not supported for Arrow type {data_type}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_extension_is_dynamic_and_clones_are_isolated() {
        let mut settings = VgiSettings::default();
        settings.set("multiplier", "3").unwrap();
        let cloned = settings.clone();
        settings.set("multiplier", "5").unwrap();
        assert_eq!(cloned.get("multiplier"), Some("3"));
        assert_eq!(settings.get("multiplier"), Some("5"));
        assert!(settings.reset_value("MULTIPLIER"));
        assert_eq!(settings.get("multiplier"), None);
    }

    #[test]
    fn struct_json_preserves_declared_child_types() {
        let fields = Fields::from(vec![
            Arc::new(Field::new("start", DataType::Int64, false)),
            Arc::new(Field::new("label", DataType::Utf8, false)),
        ]);
        let scalar = parse_struct(r#"{"start":10,"label":"item"}"#, &fields).unwrap();
        assert_eq!(scalar.data_type(), DataType::Struct(fields));
    }
}
