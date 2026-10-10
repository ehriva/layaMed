package com.convaiinnovations.laya;

/**
 * A schema that cannot be expressed as Laya questions.
 *
 * <p>The message names the <b>path</b> — {@code properties.urgency: ...} — and that is the whole
 * point of the type. A schema is refused field by field, and a caller holding a 32-property
 * schema cannot act on "this schema is not supported".
 *
 * <p>An {@link IllegalArgumentException}, matching the reference's {@code SchemaError(ValueError)}
 * and the rest of this module: a schema is an argument, and a caller catching
 * {@code IllegalArgumentException} around a call that builds one should not have to know this
 * subclass exists to catch it.
 */
public final class SchemaException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    public SchemaException(String message) {
        super(message);
    }
}
