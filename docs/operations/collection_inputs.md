# Reuse collection inputs after Drive restoration

Keep the original producer inventory, frozen configuration, preparation certificate, candidate manifest, official source pools, capability panel, legibility subset and MBPP reference report together. Preserve their original bytes and source revision. Restore the original pinned source shards and both verified tokenizer caches when a full consumer audit is needed. The source caches contain licensed data and must remain within the authorized Drive project.

Run the audit from a clean Git checkout containing both the producer and consumer commits. It accepts explicit restored paths, uses cached assets only and refuses an existing output directory. Retain the new certificate in Drive alongside the original inputs.

```bash
python scripts/revalidate_collection_inputs.py \
  --consumer-commit "$COLLECTION_CONSUMER_COMMIT" \
  --config "$COLLECTION_RUNTIME_CONFIG" \
  --producer-root "$COLLECTION_RESTORED_INPUT_ROOT" \
  --source-evidence "$COLLECTION_NATIVE_SOURCE_EVIDENCE" \
  --canonical-source-evidence "$COLLECTION_SOURCE_EVIDENCE" \
  --asset-cache "$COLLECTION_ASSET_CACHE" \
  --native-source-cache "$COLLECTION_NATIVE_SOURCE_CACHE" \
  --canonical-source-cache "$COLLECTION_CANONICAL_SOURCE_CACHE" \
  --output-dir "$COLLECTION_NEW_VALIDATION_ROOT"
```

The native and canonical cache overrides allow the restored cache paths to differ from their acquisition machine. `--project` optionally selects the clean checkout. The full manifest includes exclusions and can approach 1 GB; allow memory for JSON readback and native tokenizer batches. Progress records contain counts and identities without corpus text or credential values.

The separate `cpu_prerequisite_consumer_validation_v1` certificate binds the original producer/configuration and immutable inventory to the new consumer commit/configuration. It records exact scientific module identities, validator implementation hash, original file SHA256/size, pinned dataset/model/tokenizer identities, complete source coverage and decisions, all native token counts, fixed panel reconstruction, and the actual configured MBPP sandbox certificate. It reports no model outputs or overall launch readiness. A certificate is published only after the complete audit and final local readback checks succeed.

CPU stage reuse calls `validate_consumer_certificate(certificate_path, configuration, consumer_commit=..., input_root=..., asset_cache=...)`. That action checks the certificate's integrity/current identity, original inventory and exact restored bytes, scientific modules, both native cache manifests, panels and original MBPP report without network requests or Docker execution. Missing or incompatible evidence stops reuse. The workflow records completion only when its actual validation action succeeds.

Consumer registration can use `producer="collection_input_revalidator"` and the new `source_code_commit`, with explicit `original_producer_commit`, original payload SHA256 and consumer certificate identity. This records the new validation action while preserving the original production history. Exact-producer `recover-collection` retrieval and cross-producer input revalidation are separate operations; neither starts paid work or silently imports spending/retry state.
