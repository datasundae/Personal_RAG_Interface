# Personal RAG Interface - Project Rules

## Code Organization

### Directory Structure
```
Personal_RAG_Interface/
├── src/
│   ├── database/      # Database operations
│   ├── processing/    # Document processing
│   ├── config/        # Configuration
│   └── web/          # Web interface
├── docs/             # Documentation
├── tests/            # Test files
└── setup.py          # Package setup
```

### Module Rules
1. Each module must have its own `__init__.py`
2. Keep modules focused and single-purpose
3. Document all public interfaces
4. Use absolute imports within the package
5. Maintain backward compatibility

## Code Standards

### Python Code
1. Follow PEP 8 style guide
2. Use type hints for all function parameters and returns
3. Document all functions with docstrings
4. Keep functions focused and small
5. Use meaningful variable names
6. Handle exceptions appropriately

### Documentation
1. Keep documentation up to date
2. Include examples in docstrings
3. Document all public APIs
4. Maintain README files
5. Document configuration options

## Testing Requirements

### Unit Tests
1. Test each module independently
2. Cover all public interfaces
3. Include edge cases
4. Mock external dependencies
5. Maintain test isolation

### Integration Tests
1. Test module interactions
2. Verify data flow
3. Test error handling
4. Include performance tests
5. Test security features

## Database Operations

### Schema Changes
1. Document all schema changes
2. Provide migration scripts
3. Test migrations
4. Maintain backward compatibility
5. Version control schema files

### Data Handling
1. Validate input data
2. Sanitize user input
3. Handle sensitive data appropriately
4. Log database operations
5. Implement proper error handling

## Security Guidelines

### Data Protection
1. Encrypt sensitive data
2. Use secure connections
3. Implement access control
4. Validate all inputs
5. Log security events

### File Handling
1. Validate file types
2. Scan for malware
3. Implement size limits
4. Clean up temporary files
5. Use secure file operations

## Development Workflow

### Version Control
1. Use meaningful commit messages
2. Create feature branches
3. Review code before merging
4. Keep commits focused
5. Tag releases

### Code Review
1. Review all changes
2. Check for security issues
3. Verify documentation
4. Ensure test coverage
5. Validate performance

## Deployment

### Environment Setup
1. Document dependencies
2. Use virtual environments
3. Version control requirements
4. Document configuration
5. Provide setup scripts

### Monitoring
1. Log important events
2. Monitor performance
3. Track errors
4. Monitor security
5. Maintain backups

## Maintenance

### Updates
1. Keep dependencies current
2. Update documentation
3. Review security patches
4. Test updates thoroughly
5. Maintain changelog

### Support
1. Document known issues
2. Provide troubleshooting guides
3. Maintain issue tracker
4. Respond to bug reports
5. Update documentation 