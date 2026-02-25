# Attack Instance Examples

This directory contains intentionally vulnerable applications for security testing, training, and demonstration purposes.

⚠️ **WARNING**: All applications in this directory contain **deliberate security vulnerabilities**. They are designed for **educational purposes only** and should **NEVER** be deployed in production environments or used with real data.

## Available Examples

### 1. SQL Injection & XSS Bank (`sql_xss_bank/`)

A fake banking web application demonstrating:
- **SQL Injection** vulnerabilities in login and search functionality
- **Cross-Site Scripting (XSS)** in user-generated content
- **Weak session management**
- **Information disclosure** through error messages

**Technologies**: Python 3, Flask, SQLite

**Quick Start**:
```bash
cd sql_xss_bank
./setup.sh
./start.sh
```

See [sql_xss_bank/README.md](sql_xss_bank/README.md) for detailed documentation.

## Purpose

These examples are designed for:

- 🎓 **Security Training** - Learn about common web vulnerabilities
- 🔍 **Penetration Testing Practice** - Test your exploitation skills
- 🤖 **Tool Testing** - Validate security scanning tools like ECHO
- 📚 **Education** - Demonstrate secure vs insecure coding practices
- 🛡️ **Defensive Security** - Understand attacker techniques to build better defenses

## Ethical Use Guidelines

✅ **DO**:
- Use for educational purposes
- Test on your own systems
- Use in isolated lab environments
- Learn defensive security techniques
- Share knowledge responsibly

❌ **DO NOT**:
- Deploy to production
- Use against systems without authorization
- Store or process real user data
- Expose to the public internet
- Use for malicious purposes

## Legal Disclaimer

**IMPORTANT**: Unauthorized access to computer systems is illegal in most jurisdictions. These applications are provided solely for educational purposes in controlled environments. Users are responsible for ensuring they have proper authorization before conducting any security testing.

The authors and contributors assume no liability for misuse of these materials.

## Contributing

To add new vulnerable application examples:

1. Create a new subdirectory with a descriptive name
2. Include a comprehensive README.md documenting:
   - Vulnerabilities present
   - Test payloads and exploitation techniques
   - Setup and usage instructions
   - Security best practices that are missing
3. Provide setup scripts for easy deployment
4. Update this README with a link to your example

## Security Notice

If you discover unintended vulnerabilities or security issues with the example applications themselves, please report them responsibly to the maintainers.

---

**Remember**: With great power comes great responsibility. Use these tools ethically and legally.
