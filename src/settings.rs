// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! DataFusion-native VGI session settings.
//!
//! DataFusion requires third-party settings to live under a configuration
//! namespace. `VgiSettings` therefore exposes `vgi.<setting>` while worker
//! settings retain their original unprefixed names on the wire. A small fixed
//! set of adapter-owned tuning settings live in the same extension but are
//! only forwarded when a worker independently declares a setting of that name.

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VgiSettings {
    values: BTreeMap<String, String>,
}

pub(crate) const EXCHANGE_INPUT_DEDUP: &str = "vgi_exchange_input_dedup";
pub(crate) const RESULT_CACHE_PER_VALUE: &str = "vgi_result_cache_per_value";

const ADAPTER_BOOLEAN_SETTINGS: [&str; 2] = [EXCHANGE_INPUT_DEDUP, RESULT_CACHE_PER_VALUE];

/// Typed snapshot of settings consumed by the DataFusion adapter itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VgiAdapterSettings {
    exchange_input_dedup: bool,
    result_cache_per_value: bool,
}

impl VgiAdapterSettings {
    pub(crate) fn exchange_input_dedup(self) -> bool {
        self.exchange_input_dedup
    }

    pub(crate) fn result_cache_per_value(self) -> bool {
        self.result_cache_per_value
    }
}

impl Default for VgiSettings {
    fn default() -> Self {
        Self {
            values: ADAPTER_BOOLEAN_SETTINGS
                .into_iter()
                .map(|name| (name.to_string(), "true".to_string()))
                .collect(),
        }
    }
}

impl VgiSettings {
    /// Set one VGI setting without going through SQL.
    pub fn set_value(&mut self, name: impl Into<String>, value: impl Into<String>) -> DFResult<()> {
        let name = normalize_name(&name.into())?;
        let value = value.into();
        let value = if is_adapter_setting(&name) {
            parse_bool(&name, &value)?.to_string()
        } else {
            value
        };
        self.values.insert(name, value);
        Ok(())
    }

    /// Reset a setting to its default.
    ///
    /// Dynamic worker settings are removed. Adapter-owned booleans are
    /// restored to `true`.
    pub fn reset_value(&mut self, name: &str) -> bool {
        let name = name.to_ascii_lowercase();
        if is_adapter_setting(&name) {
            return self.values.insert(name, "true".to_string()).is_some();
        }
        self.values.remove(&name).is_some()
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

    pub(crate) fn adapter_settings(&self) -> VgiAdapterSettings {
        VgiAdapterSettings {
            exchange_input_dedup: self.adapter_bool(EXCHANGE_INPUT_DEDUP),
            result_cache_per_value: self.adapter_bool(RESULT_CACHE_PER_VALUE),
        }
    }

    fn adapter_bool(&self, name: &str) -> bool {
        // `set_value` validates and canonicalizes adapter booleans. Falling
        // back to true also keeps hand-built/default snapshots conservative.
        self.get(name)
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(true)
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
                description: if is_adapter_setting(name) {
                    "DataFusion VGI adapter tuning; forwarded only when the worker declares the same setting"
                } else {
                    "Value forwarded using the attached VGI worker's declared Arrow type"
                },
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

pub(crate) fn is_adapter_setting(name: &str) -> bool {
    ADAPTER_BOOLEAN_SETTINGS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(name))
}

fn parse_bool(name: &str, value: &str) -> DFResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "1" => Ok(true),
        "false" | "off" | "0" => Ok(false),
        _ => Err(DataFusionError::Configuration(format!(
            "VGI adapter setting `{name}` expects a boolean, found `{value}`"
        ))),
    }
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
    fn adapter_settings_are_typed_default_true_and_reset_to_default() {
        let mut settings = VgiSettings::default();
        assert_eq!(
            settings.adapter_settings(),
            VgiAdapterSettings {
                exchange_input_dedup: true,
                result_cache_per_value: true,
            }
        );
        let keys = settings
            .entries()
            .into_iter()
            .map(|entry| entry.key)
            .collect::<Vec<_>>();
        assert!(keys.contains(&format!("vgi.{EXCHANGE_INPUT_DEDUP}")));
        assert!(keys.contains(&format!("vgi.{RESULT_CACHE_PER_VALUE}")));

        settings.set_value(EXCHANGE_INPUT_DEDUP, "OFF").unwrap();
        settings.set_value(RESULT_CACHE_PER_VALUE, "false").unwrap();
        assert!(!settings.adapter_settings().exchange_input_dedup());
        assert!(!settings.adapter_settings().result_cache_per_value());

        assert!(settings.reset_value(EXCHANGE_INPUT_DEDUP));
        assert!(settings.reset_value(RESULT_CACHE_PER_VALUE));
        assert!(settings.adapter_settings().exchange_input_dedup());
        assert!(settings.adapter_settings().result_cache_per_value());
        assert!(settings.set_value(EXCHANGE_INPUT_DEDUP, "maybe").is_err());
    }

    #[test]
    fn adapter_settings_are_only_encoded_for_matching_worker_declarations() {
        let mut settings = VgiSettings::default();
        settings.set_value(RESULT_CACHE_PER_VALUE, "false").unwrap();
        assert!(encode_settings(&settings, &[]).unwrap().is_none());

        let declaration = vgi_client::SettingSpec {
            name: RESULT_CACHE_PER_VALUE.to_string(),
            description: "worker independently declares the same setting".to_string(),
            data_type: DataType::Boolean,
            default_value: None,
        };
        let encoded = encode_settings(&settings, &[declaration])
            .unwrap()
            .expect("matching worker declaration is forwarded");
        let batch = vgi_protocol::ipc::read_batch(&encoded.0).unwrap();
        let values = batch
            .column_by_name(RESULT_CACHE_PER_VALUE)
            .unwrap()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::BooleanArray>()
            .unwrap();
        assert!(!values.value(0));
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
